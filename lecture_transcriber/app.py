"""Desktop app: native window (pywebview / WebKit) around the local UI.

Started by Vorlesung.app (see install_mac.sh) or with `vorlesung app`.
Falls back to the default browser if pywebview isn't installed.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from .engine import Engine

DEFAULT_OUTPUT = str(Path.home() / "Vorlesungen")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vorlesung app", description="Vorlesung als App starten")
    p.add_argument("-o", "--output", default=DEFAULT_OUTPUT, help="Ordner für Transkripte")
    p.add_argument("-m", "--model", default=None, help="Whisper-Modell (Standard: automatisch)")
    p.add_argument("-l", "--language", default="de", help="Sprache (Standard: de)")
    p.add_argument("--backend", default="auto", choices=["auto", "mlx", "faster-whisper"])
    p.add_argument("--browser", action="store_true", help="Im Browser statt im App-Fenster öffnen")
    p.add_argument("--host", default="127.0.0.1", help=argparse.SUPPRESS)
    p.add_argument("--port", type=int, default=8765, help=argparse.SUPPRESS)
    p.add_argument("--no-open", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--input-file", default=None, help=argparse.SUPPRESS)  # testing
    p.add_argument("--speed", type=float, default=1.0, help=argparse.SUPPRESS)
    return p


def _save_dialog(webview, window, filename: str):
    """Native macOS "Save as" sheet. Works with pywebview 5 and 6."""
    kind = getattr(getattr(webview, "FileDialog", None), "SAVE", None)
    if kind is None:
        kind = webview.SAVE_DIALOG
    result = window.create_file_dialog(
        kind, directory=str(Path.home() / "Documents"), save_filename=filename
    )
    if isinstance(result, (list, tuple)):
        result = result[0] if result else None
    return result or None


def _mac_branding():
    """Show "Vorlesung" and the app icon in the Dock/menu instead of "Python"."""
    if sys.platform != "darwin":
        return
    try:
        from Foundation import NSBundle

        info = NSBundle.mainBundle().infoDictionary()
        info["CFBundleName"] = "Vorlesung"
        icon = os.environ.get("VORLESUNG_ICON")
        if icon and os.path.exists(icon):
            from AppKit import NSApplication, NSImage

            image = NSImage.alloc().initWithContentsOfFile_(icon)
            NSApplication.sharedApplication().setApplicationIconImage_(image)
    except Exception:
        pass


def run_app(args) -> int:
    from .server import AppServer

    engine = Engine(args.output, model=args.model, language=args.language, backend=args.backend)
    engine.load_model_async()
    server = AppServer(engine, host=args.host, port=args.port,
                       test_input=args.input_file, test_speed=args.speed)
    server.start()
    print(f"Vorlesung läuft auf {server.url}", flush=True)

    webview = None
    if not args.browser:
        try:
            import webview  # pywebview
        except ImportError:
            webview = None

    try:
        if webview is not None:
            window = webview.create_window(
                "Vorlesung", server.url, width=1120, height=760, min_size=(640, 480),
                text_select=True,
            )

            def on_closing():
                # Closing the window while recording: stop and save cleanly.
                if engine.busy:
                    engine.stop()
                return True

            window.events.closing += on_closing
            server.save_dialog = lambda name: _save_dialog(webview, window, name)
            _mac_branding()
            try:
                # Not private: keeps remembered settings (microphone, terms) between starts.
                webview.start(private_mode=False)
            except TypeError:  # older pywebview
                webview.start()
        else:
            if not args.no_open:
                import webbrowser

                webbrowser.open(server.url)
            print("Beenden mit Strg+C", flush=True)
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        if engine.busy:
            print("Speichere die laufende Aufnahme…", flush=True)
            engine.stop()
            try:
                engine.wait()
            except KeyboardInterrupt:
                pass
        server.close()
    return 0


def main(argv=None) -> int:
    return run_app(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
