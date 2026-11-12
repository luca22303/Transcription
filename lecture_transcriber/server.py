"""Local HTTP server that serves the app UI and talks to the engine.

Stdlib only. The UI receives live updates via Server-Sent Events.
Bound to 127.0.0.1 by default. POST requests need a custom header and the
Host header is checked, so other websites can't remote-control the app.
"""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Callable, List, Optional
from urllib.parse import parse_qs, quote, unquote, urlparse

from .engine import Engine
from .export import FORMATS, available_formats, default_filename, export_bytes, export_to


def _ui_html() -> bytes:
    return resources.files("lecture_transcriber").joinpath("ui.html").read_bytes()


def reveal_in_finder(path: Path):
    if sys.platform == "darwin":
        subprocess.Popen(["open", "-R", str(path)])
    elif sys.platform.startswith("win"):
        subprocess.Popen(["explorer", "/select,", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path.parent)])


class AppServer:
    def __init__(self, engine: Engine, host: str = "127.0.0.1", port: int = 8765,
                 test_input: Optional[str] = None, test_speed: float = 1.0):
        self.engine = engine
        self.test_input = test_input
        self.test_speed = test_speed
        # Set by the desktop app: shows a native "Save as" dialog and returns
        # the chosen path (or None if cancelled).
        self.save_dialog: Optional[Callable[[str], Optional[str]]] = None
        self._clients: List[queue.Queue] = []
        self._lock = threading.Lock()
        engine.on(self._broadcast)
        handler = self._make_handler()
        try:
            self.httpd = ThreadingHTTPServer((host, port), handler)
        except OSError:
            # Port taken (e.g. app already running) -> pick any free port.
            self.httpd = ThreadingHTTPServer((host, 0), handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.open_to_network = host not in ("127.0.0.1", "localhost", "::1")
        self.url = f"http://127.0.0.1:{self.port}/"

    def start(self):
        threading.Thread(target=self.httpd.serve_forever, daemon=True, name="http").start()

    def close(self):
        self.httpd.shutdown()

    def _broadcast(self, event: str, data: dict):
        with self._lock:
            for q in self._clients:
                q.put((event, data))

    # -- request handling -------------------------------------------------------
    def _make_handler(self):
        app = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            # helpers
            def _host_ok(self) -> bool:
                if app.open_to_network:
                    return True
                host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
                return host in ("127.0.0.1", "localhost", "::1")

            def _json(self, data, status=200):
                body = json.dumps(data, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _error(self, message, status=400):
                self._json({"error": message}, status)

            def _read_json(self) -> dict:
                # Always consume the body, otherwise leftover bytes corrupt the
                # next request on this keep-alive connection.
                length = int(self.headers.get("Content-Length") or 0)
                if not length:
                    return {}
                raw = self.rfile.read(length)
                try:
                    data = json.loads(raw.decode("utf-8"))
                except ValueError:
                    return {}
                return data if isinstance(data, dict) else {}

            # routes
            def do_GET(self):
                if not self._host_ok():
                    return self._error("forbidden", 403)
                url = urlparse(self.path)
                path = url.path
                if path == "/":
                    body = _ui_html()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                elif path == "/events":
                    self._events()
                elif path == "/api/devices":
                    self._devices()
                elif path == "/api/sessions":
                    self._json(app.engine.list_sessions())
                elif path.startswith("/api/sessions/") and path.endswith("/export"):
                    self._download(unquote(path[len("/api/sessions/"):-len("/export")]),
                                   parse_qs(url.query).get("format", ["docx"])[0])
                elif path.startswith("/api/sessions/"):
                    data = app.engine.load_session(unquote(path[len("/api/sessions/"):]))
                    if data is None:
                        return self._error("not found", 404)
                    self._json(data)
                else:
                    self._error("not found", 404)

            def do_POST(self):
                url = urlparse(self.path)
                if not self._host_ok() or self.headers.get("X-Vorlesung") != "1":
                    self.close_connection = True
                    return self._error("forbidden", 403)
                if url.path == "/api/upload":
                    try:
                        return self._upload(parse_qs(url.query))
                    except (RuntimeError, SystemExit, OSError) as exc:
                        self.close_connection = True
                        return self._error(str(exc), 409)
                body = self._read_json()
                try:
                    if url.path == "/api/start":
                        return self._start(body)
                    if url.path == "/api/stop":
                        app.engine.stop()
                        return self._json({"ok": True})
                    if url.path == "/api/cancel":
                        app.engine.cancel()
                        return self._json({"ok": True})
                    if url.path == "/api/save":
                        return self._save(body)
                    if url.path == "/api/reveal":
                        path = app.engine.session_path(str(body.get("id", "")))
                        if path is None:
                            return self._error("not found", 404)
                        target = path / "transkript.md"
                        reveal_in_finder(target if target.exists() else path)
                        return self._json({"ok": True})
                except (RuntimeError, SystemExit, OSError) as exc:
                    return self._error(str(exc), 409)
                self._error("not found", 404)

            def _export_args(self, session_id: str, fmt: str):
                data = app.engine.load_session(session_id)
                if data is None:
                    return None
                session_dir = Path(data["path"])
                if fmt not in available_formats(session_dir):
                    return None
                return session_dir, data["info"], data["paragraphs"]

            def _download(self, session_id: str, fmt: str):
                args = self._export_args(session_id, fmt)
                if args is None:
                    return self._error("not found", 404)
                session_dir, info, paragraphs = args
                name = default_filename(info, fmt)
                if fmt == "wav":
                    body = (session_dir / "aufnahme.wav").read_bytes()
                else:
                    body = export_bytes(session_dir, info, paragraphs, fmt)
                self.send_response(200)
                self.send_header("Content-Type", FORMATS[fmt]["mime"])
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Content-Disposition",
                                 f"attachment; filename=\"transkript{FORMATS[fmt]['ext']}\"; "
                                 f"filename*=UTF-8''{quote(name)}")
                self.end_headers()
                self.wfile.write(body)

            def _save(self, body: dict):
                fmt = str(body.get("format", "docx"))
                args = self._export_args(str(body.get("id", "")), fmt)
                if args is None:
                    return self._error("not found", 404)
                if app.save_dialog is None:
                    return self._error("no dialog", 501)  # UI falls back to a download
                session_dir, info, paragraphs = args
                chosen = app.save_dialog(default_filename(info, fmt))
                if not chosen:
                    return self._json({"ok": False, "cancelled": True})
                target = export_to(session_dir, info, paragraphs, fmt, Path(chosen))
                return self._json({"ok": True, "path": str(target)})

            def _devices(self):
                error = ""
                try:
                    from .audio import input_devices

                    devices = input_devices()
                except BaseException as exc:
                    devices, error = [], str(exc)
                if app.test_input:
                    devices.insert(0, {"id": "test", "name": "Testdatei", "default": True})
                    error = ""
                self._json({"devices": devices, "error": error})

            def _start(self, data: dict):
                source = None
                device = data.get("device") or None
                if app.test_input and device in (None, "test"):
                    from .audio import FileSource

                    source = FileSource(app.test_input, speed=app.test_speed)
                session = app.engine.start_live(
                    title=(data.get("title") or "").strip(),
                    device=device,
                    prompt=(data.get("prompt") or "").strip(),
                    source=source,
                    eco=bool(data.get("eco")),
                )
                self._json({"ok": True, "session": session.name})

            def _upload(self, query):
                name = Path(unquote(query.get("filename", ["aufnahme"])[0])).name or "aufnahme"
                title = (query.get("title", [""])[0]).strip()
                prompt = (query.get("prompt", [""])[0]).strip()
                length = int(self.headers.get("Content-Length") or 0)
                if length <= 0:
                    return self._error("Leere Datei")
                if app.engine.busy:
                    self.close_connection = True  # body is not read
                    return self._error("Es läuft bereits eine Aufnahme oder Transkription.", 409)
                tmp_dir = app.engine.output_dir / ".import"
                tmp_dir.mkdir(exist_ok=True)
                tmp = tmp_dir / name
                remaining = length
                with open(tmp, "wb") as fh:
                    while remaining > 0:
                        block = self.rfile.read(min(remaining, 1 << 20))
                        if not block:
                            break
                        fh.write(block)
                        remaining -= len(block)
                session = app.engine.transcribe_file(
                    str(tmp), title=title or Path(name).stem, prompt=prompt, move=True
                )
                self._json({"ok": True, "session": session.name})

            def _events(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                q: queue.Queue = queue.Queue()
                with app._lock:
                    app._clients.append(q)
                    init = app.engine.snapshot()
                    init["native_save"] = app.save_dialog is not None
                try:
                    self._send("init", init)
                    while True:
                        try:
                            event, data = q.get(timeout=15)
                        except queue.Empty:
                            self.wfile.write(b": ping\n\n")
                            self.wfile.flush()
                            continue
                        self._send(event, data)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    with app._lock:
                        if q in app._clients:
                            app._clients.remove(q)
                    self.close_connection = True

            def _send(self, event, data):
                payload = json.dumps(data, ensure_ascii=False)
                self.wfile.write(f"event: {event}\ndata: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()

        return Handler
