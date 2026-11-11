"""The engine: owns the speech model and runs one job at a time.

Jobs are either a live recording (microphone -> split at pauses ->
transcribe) or the transcription of an existing audio/video file. Both are
written incrementally to a session folder. The UI and the CLI subscribe to
events via `Engine.on(callback)`.

Recording and transcription run in separate threads, so audio is never
dropped even if recognition briefly falls behind. There is no time limit.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

from .audio import SAMPLE_RATE, FileSource, MicrophoneSource, decode_audio
from .output import TranscriptWriter, WavRecorder, fmt_clock, new_session_dir
from .segmenter import Chunk, Segmenter
from .transcriber import Segment, Transcriber

Listener = Callable[[str, dict], None]


def chunks_of_file(audio: np.ndarray) -> Iterator[Chunk]:
    """Split a whole recording at pauses, exactly like the live pipeline does."""
    seg = Segmenter()
    block = SAMPLE_RATE  # feed 1 s blocks
    for i in range(0, len(audio), block):
        yield from seg.push(audio[i : i + block])
    yield from seg.flush()


class KeepAwake:
    """Keeps the Mac awake and the app at full speed while a job runs.

    * `caffeinate` prevents idle sleep.
    * An NSProcessInfo activity stops macOS from throttling the app
      ("App Nap") while its window is in the background, e.g. while you
      take notes in another app during a long lecture.
    """

    def __init__(self):
        self._proc = None
        self._activity = None

    def __enter__(self):
        if sys.platform == "darwin":
            try:
                self._proc = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
            except OSError:
                self._proc = None
            try:
                from Foundation import NSProcessInfo

                user_initiated = 0x00FFFFFF | (1 << 20)  # NSActivityUserInitiated
                latency_critical = 0xFF00000000  # NSActivityLatencyCritical
                self._activity = NSProcessInfo.processInfo().beginActivityWithOptions_reason_(
                    user_initiated | latency_critical, "Vorlesung wird aufgenommen"
                )
            except Exception:  # pyobjc missing (CLI-only install)
                self._activity = None
        return self

    def __exit__(self, *exc):
        if self._proc is not None:
            self._proc.terminate()
            self._proc = None
        if self._activity is not None:
            try:
                from Foundation import NSProcessInfo

                NSProcessInfo.processInfo().endActivity_(self._activity)
            except Exception:
                pass
            self._activity = None


class Engine:
    def __init__(
        self,
        output_dir: str,
        model: Optional[str] = None,
        language: str = "de",
        backend: str = "auto",
        beam_size: int = 5,
    ):
        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._model_args = dict(model=model, language=language, backend=backend, beam_size=beam_size)
        self._listeners: List[Listener] = []
        self._transcriber: Optional[Transcriber] = None
        self._model_ready = threading.Event()
        self._model_error: Optional[str] = None
        self._lock = threading.RLock()
        self._job_thread: Optional[threading.Thread] = None
        self._source = None
        self._cancel = threading.Event()
        self.paragraphs: List[dict] = []
        self.state: Dict = {
            "mode": "idle",  # idle | recording | finishing | file
            "model": "loading",  # loading | ready | error
            "model_desc": "",
            "model_error": "",
            "title": "",
            "session": "",
            "elapsed": 0.0,
            "level": -90.0,
            "backlog": 0,
            "progress": 0.0,
            "warning": "",
            "last_session": "",
        }

    # -- events -------------------------------------------------------------
    def on(self, listener: Listener):
        self._listeners.append(listener)

    def _emit(self, event: str, data: dict):
        for listener in list(self._listeners):
            try:
                listener(event, data)
            except Exception:
                pass

    def _update(self, **changes):
        with self._lock:
            self.state.update(changes)
            snapshot = dict(self.state)
        self._emit("state", snapshot)

    def snapshot(self) -> dict:
        with self._lock:
            return {"state": dict(self.state), "paragraphs": list(self.paragraphs)}

    # -- model --------------------------------------------------------------
    def load_model_async(self):
        def load():
            try:
                tr = Transcriber(**self._model_args)
            except BaseException as exc:  # SystemExit from missing backends, too
                self._model_error = str(exc) or exc.__class__.__name__
                self._update(model="error", model_error=self._model_error)
            else:
                self._transcriber = tr
                self._update(model="ready", model_desc=tr.description)
            finally:
                self._model_ready.set()

        threading.Thread(target=load, daemon=True, name="model-loader").start()

    def _wait_model(self) -> Transcriber:
        self._model_ready.wait()
        if self._transcriber is None:
            raise RuntimeError(self._model_error or "Modell konnte nicht geladen werden")
        return self._transcriber

    # -- jobs ---------------------------------------------------------------
    @property
    def busy(self) -> bool:
        return self._job_thread is not None and self._job_thread.is_alive()

    def _new_session(self, title: str, kind: str, extra: Optional[dict] = None) -> Path:
        session = new_session_dir(str(self.output_dir), title)
        info = {
            "title": title or ("Vorlesung" if kind == "live" else "Aufnahme"),
            "started": datetime.now().isoformat(timespec="seconds"),
            "kind": kind,
            "duration": 0,
        }
        info.update(extra or {})
        (session / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), "utf-8")
        with self._lock:
            self.paragraphs = []
        self._emit("reset", {"title": info["title"], "session": session.name})
        return session

    @staticmethod
    def _finish_info(session: Path, **fields):
        path = session / "info.json"
        try:
            info = json.loads(path.read_text("utf-8"))
        except Exception:
            info = {}
        info.update(fields)
        path.write_text(json.dumps(info, ensure_ascii=False, indent=2), "utf-8")

    def _publish(self, writer: TranscriptWriter, segments: List[Segment]):
        if not segments:
            return
        writer.add_paragraph(segments)
        item = {"time": fmt_clock(segments[0].start), "text": " ".join(s.text for s in segments)}
        with self._lock:
            self.paragraphs.append(item)
        self._emit("paragraph", item)

    def start_live(
        self,
        title: str = "",
        device: Optional[str] = None,
        prompt: str = "",
        source=None,
        save_audio: bool = True,
    ) -> Path:
        """Start recording. Raises SystemExit/RuntimeError if the device can't be opened."""
        with self._lock:
            if self.busy:
                raise RuntimeError("Es läuft bereits eine Aufnahme oder Transkription.")
            src = source or MicrophoneSource(device)
            src.start()  # fail early (permissions, unknown device) before creating files
            self._source = src
            self._cancel.clear()
            session = self._new_session(title, "live", {"source": src.name})
            self._job_thread = threading.Thread(
                target=self._live_job, args=(session, title, prompt, src, save_audio),
                daemon=True, name="live-job",
            )
            self._update(mode="recording", title=title, session=session.name, elapsed=0.0,
                         backlog=0, warning="", progress=0.0)
            self._job_thread.start()
            return session

    def stop(self):
        """Stop the recording. Remaining audio is still transcribed."""
        src = self._source
        if src is not None:
            src.stop()

    def cancel(self):
        """Stop and skip the transcription of anything not yet processed."""
        self._cancel.set()
        self.stop()

    def wait(self, timeout: Optional[float] = None):
        t = self._job_thread
        if t is not None:
            t.join(timeout)

    def _live_job(self, session: Path, title: str, prompt: str, src, save_audio: bool):
        writer = TranscriptWriter(session, title)
        recorder = WavRecorder(session / "aufnahme.wav") if save_audio else None
        segmenter = Segmenter()
        chunks: "queue.Queue[Optional[Chunk]]" = queue.Queue()
        stats = {"audio": 0.0, "compute": 0.0}

        def worker():
            try:
                tr = self._wait_model()
            except RuntimeError as exc:
                self._update(warning=f"Spracherkennung nicht verfügbar: {exc}. "
                                     "Audio wird trotzdem aufgenommen.")
                return
            tr.reset(prompt)
            writer.set_note(tr.description)
            while True:
                chunk = chunks.get()
                if chunk is None or self._cancel.is_set():
                    return
                if not chunk.has_speech:
                    continue
                t = time.monotonic()
                try:
                    segments = tr.transcribe_chunk(chunk.audio, chunk.start)
                except Exception as exc:  # keep going, never lose the lecture
                    self._emit("log", {"text": f"Abschnitt übersprungen: {exc}"})
                    continue
                stats["compute"] += time.monotonic() - t
                stats["audio"] += len(chunk.audio) / SAMPLE_RATE
                self._publish(writer, segments)

        worker_thread = threading.Thread(target=worker, daemon=True, name="transcribe")
        worker_thread.start()

        recorded = 0
        peak = 0.0
        recent: List[np.ndarray] = []
        last_status = 0.0
        reported_drops = 0
        with KeepAwake():
            for block in src.blocks():
                if recorder:
                    recorder.write(block)
                recorded += len(block)
                peak = max(peak, float(np.max(np.abs(block))) if len(block) else 0.0)
                recent.append(block)
                for chunk in segmenter.push(block):
                    chunks.put(chunk)
                now = time.monotonic()
                if now - last_status >= 0.4:
                    last_status = now
                    audio = np.concatenate(recent) if recent else np.zeros(1)
                    recent = []
                    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)) + 1e-9)
                    elapsed = recorded / SAMPLE_RATE
                    warning = self.state.get("warning", "")
                    if elapsed > 4 and peak == 0.0:
                        warning = ("Kein Ton vom Mikrofon. Mikrofon-Zugriff erlauben: "
                                   "Systemeinstellungen › Datenschutz & Sicherheit › Mikrofon.")
                    elif warning.startswith("Kein Ton"):
                        warning = ""
                    drops = getattr(src, "dropouts", 0)
                    if drops > reported_drops:
                        reported_drops = drops
                        warning = ("Audio-Aussetzer erkannt: Der Mac kam mit der Aufnahme kurz nicht "
                                   "hinterher. Andere rechenintensive Programme schließen hilft.")
                    backlog = chunks.qsize()
                    if backlog >= 3 and self._model_ready.is_set() and self._transcriber:
                        warning = ("Die Erkennung kommt nicht hinterher – der Text hängt etwas "
                                   "hinterher, die Aufnahme ist aber vollständig.")
                    elif warning.startswith("Die Erkennung") and backlog == 0:
                        warning = ""
                    self._update(elapsed=elapsed, level=20 * np.log10(rms), backlog=backlog,
                                 warning=warning)

            self._source = None
            duration = recorded / SAMPLE_RATE
            self._update(mode="finishing", elapsed=duration, level=-90.0)
            for chunk in segmenter.flush():
                chunks.put(chunk)
            chunks.put(None)
            while worker_thread.is_alive():
                worker_thread.join(0.3)
                self._update(backlog=chunks.qsize())

        if recorder:
            recorder.close()
        writer.close()
        speed = stats["audio"] / stats["compute"] if stats["compute"] else 0
        self._finish_info(session, duration=round(duration, 1),
                          model=self._transcriber.description if self._transcriber else "")
        self._update(mode="idle", backlog=0, last_session=session.name, warning="")
        self._emit("finished", {"session": session.name, "path": str(session),
                                "duration": duration, "speed": speed,
                                "transcript": str(writer.md_path)})

    def transcribe_file(self, path: str, title: str = "", prompt: str = "",
                        move: bool = False) -> Path:
        """Transcribe an existing recording into a new session folder."""
        src = Path(path)
        with self._lock:
            if self.busy:
                raise RuntimeError("Es läuft bereits eine Aufnahme oder Transkription.")
            self._cancel.clear()
            session = self._new_session(title or src.stem, "datei", {"source": src.name})
            target = session / src.name
            if move:
                shutil.move(str(src), target)
            else:
                target = src
            self._job_thread = threading.Thread(
                target=self._file_job, args=(session, title or src.stem, prompt, target),
                daemon=True, name="file-job",
            )
            self._update(mode="file", title=title or src.stem, session=session.name,
                         progress=0.0, elapsed=0.0, warning="")
            self._job_thread.start()
            return session

    def _file_job(self, session: Path, title: str, prompt: str, path: Path):
        writer = TranscriptWriter(session, title)
        duration = 0.0
        failure = ""
        try:
            with KeepAwake():
                self._update(warning="Lese Datei…")
                audio = decode_audio(str(path))
                duration = len(audio) / SAMPLE_RATE
                self._update(warning="" if self._transcriber else "Warte auf Modell…")
                tr = self._wait_model()
                tr.reset(prompt)
                writer.set_note(tr.description)
                self._update(warning="")
                for chunk in chunks_of_file(audio):
                    if self._cancel.is_set():
                        break
                    if chunk.has_speech:
                        self._publish(writer, tr.transcribe_chunk(chunk.audio, chunk.start))
                    self._update(progress=min(1.0, chunk.end / max(duration, 1e-6)),
                                 elapsed=chunk.end)
        except BaseException as exc:
            failure = f"Datei konnte nicht transkribiert werden: {exc}"
            self._emit("failure", {"message": failure})
        finally:
            writer.close()
            self._finish_info(session, duration=round(duration, 1),
                              model=self._transcriber.description if self._transcriber else "")
            self._update(mode="idle", progress=1.0, last_session=session.name, warning=failure)
            self._emit("finished", {"session": session.name, "path": str(session),
                                    "duration": duration, "transcript": str(writer.md_path)})

    # -- archive ------------------------------------------------------------
    def list_sessions(self) -> List[dict]:
        items = []
        for d in self.output_dir.iterdir() if self.output_dir.exists() else []:
            info_path = d / "info.json"
            if not d.is_dir() or not info_path.exists():
                continue
            try:
                info = json.loads(info_path.read_text("utf-8"))
            except Exception:
                continue
            info["id"] = d.name
            items.append(info)
        items.sort(key=lambda i: i.get("started", ""), reverse=True)
        return items

    def session_path(self, session_id: str) -> Optional[Path]:
        if not session_id or "/" in session_id or "\\" in session_id or session_id.startswith("."):
            return None
        path = self.output_dir / session_id
        return path if (path / "info.json").exists() else None

    def load_session(self, session_id: str) -> Optional[dict]:
        path = self.session_path(session_id)
        if path is None:
            return None
        info = json.loads((path / "info.json").read_text("utf-8"))
        info["id"] = session_id
        paragraphs = []
        md = path / "transkript.md"
        if md.exists():
            for line in md.read_text("utf-8").splitlines():
                if line.startswith("**[") and "]** " in line:
                    stamp, text = line[3:].split("]** ", 1)
                    paragraphs.append({"time": stamp, "text": text})
        return {"info": info, "paragraphs": paragraphs, "path": str(path)}
