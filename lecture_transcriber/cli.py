"""Command line interface.

    vorlesung                 -> opens the app (same as `vorlesung app`)
    vorlesung live "Titel"    -> records in the terminal, Ctrl+C to stop
    vorlesung datei aufnahme.m4a
    vorlesung geraete
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

DEFAULT_OUTPUT = str(Path.home() / "Vorlesungen")


def _add_model_args(p: argparse.ArgumentParser):
    p.add_argument(
        "-m", "--model", default=None,
        help="Whisper-Modell: tiny, base, small, medium, large-v3, large-v3-turbo "
             "(Standard: large-v3-turbo auf Apple Silicon/NVIDIA, sonst small)",
    )
    p.add_argument("-l", "--language", default="de",
                   help="Sprache, z.B. de, en, oder 'auto' (Standard: de)")
    p.add_argument("--backend", default="auto", choices=["auto", "mlx", "faster-whisper"],
                   help="auto = MLX (Apple-GPU) auf M-Macs, sonst faster-whisper")
    p.add_argument("-p", "--prompt", default="",
                   help="Fachbegriffe, z.B. 'Eigenwert, Determinante, Gauß-Algorithmus'")
    p.add_argument("-o", "--output", default=DEFAULT_OUTPUT,
                   help=f"Ordner für Transkripte (Standard: {DEFAULT_OUTPUT})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vorlesung",
        description="Vorlesungen live und ohne Zeitlimit lokal transkribieren.",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("app", help="App-Fenster öffnen (Standard)", add_help=False)

    live = sub.add_parser("live", help="Im Terminal aufnehmen und live mitschreiben")
    live.add_argument("titel", nargs="?", default="", help="Name der Vorlesung, z.B. 'Analysis I'")
    _add_model_args(live)
    live.add_argument("-d", "--device", default=None,
                      help="Eingabegerät (Nummer oder Teil des Namens, siehe `vorlesung geraete`)")
    live.add_argument("--no-audio", action="store_true", help="Audio nicht als WAV speichern")
    live.add_argument("--input-file", default=None, help=argparse.SUPPRESS)  # testing
    live.add_argument("--speed", type=float, default=1.0, help=argparse.SUPPRESS)

    datei = sub.add_parser("datei", aliases=["file"],
                           help="Aufnahme/Video (MP3, M4A, WAV, MP4…) transkribieren")
    datei.add_argument("pfad", help="Audio- oder Videodatei")
    datei.add_argument("-t", "--titel", default="", help="Name der Vorlesung")
    _add_model_args(datei)

    sub.add_parser("geraete", aliases=["devices"], help="Verfügbare Mikrofone/Eingänge anzeigen")
    return parser


def _console(engine, done: threading.Event):
    from .output import fmt_clock

    state = {"warning": "", "model": "loading"}

    def listener(event: str, data: dict):
        if event == "paragraph":
            print(f"\n[{data['time']}] {data['text']}", flush=True)
        elif event == "state":
            if data["model"] != state["model"]:
                state["model"] = data["model"]
                if data["model"] == "ready":
                    print(f"Spracherkennung bereit: {data['model_desc']}", flush=True)
                elif data["model"] == "error":
                    print(f"Spracherkennung nicht verfügbar: {data['model_error']}", flush=True)
            if data["warning"] != state["warning"]:
                state["warning"] = data["warning"]
                if data["warning"]:
                    print(f"\n[Hinweis: {data['warning']}]", flush=True)
            if data["mode"] == "file":
                pct = int(data["progress"] * 100) // 10 * 10
                if pct != state.get("pct"):
                    state["pct"] = pct
                    print(f"  … {pct} %", flush=True)
        elif event == "failure":
            print(f"\nFehler: {data['message']}", flush=True)
        elif event == "finished":
            print(f"\nFertig – Dauer {fmt_clock(data['duration'])}.", flush=True)
            if data.get("speed"):
                print(f"Erkennung lief {data['speed']:.1f}x schneller als Echtzeit.", flush=True)
            print(f"Transkript: {data['transcript']}", flush=True)
            done.set()

    return listener


def run_live(args) -> int:
    from .engine import Engine

    engine = Engine(args.output, model=args.model, language=args.language, backend=args.backend)
    done = threading.Event()
    engine.on(_console(engine, done))
    print("Lade Spracherkennung (beim ersten Mal wird das Modell heruntergeladen)…", flush=True)
    engine.load_model_async()

    source = None
    if args.input_file:
        from .audio import FileSource

        source = FileSource(args.input_file, speed=args.speed)
    session = engine.start_live(title=args.titel, device=args.device, prompt=args.prompt,
                                source=source, save_audio=not args.no_audio)
    print(f"\n● Aufnahme läuft – {session}\n  Beenden mit Strg+C\n", flush=True)
    try:
        while not done.wait(0.5):
            pass
    except KeyboardInterrupt:
        print("\n■ Aufnahme beendet, verarbeite den Rest… (nochmal Strg+C = sofort beenden)", flush=True)
        engine.stop()
        try:
            done.wait()
        except KeyboardInterrupt:
            engine.cancel()
            engine.wait(5)
    return 0


def run_file(args) -> int:
    from .engine import Engine

    src = Path(args.pfad)
    if not src.exists():
        raise SystemExit(f"Datei nicht gefunden: {src}")
    engine = Engine(args.output, model=args.model, language=args.language, backend=args.backend)
    done = threading.Event()
    engine.on(_console(engine, done))
    print("Lade Spracherkennung…", flush=True)
    engine.load_model_async()
    engine.transcribe_file(str(src), title=args.titel or src.stem, prompt=args.prompt)
    try:
        done.wait()
    except KeyboardInterrupt:
        engine.cancel()
        engine.wait(5)
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] == "app":
        from .app import main as app_main

        return app_main(argv[1:])

    args = build_parser().parse_args(argv)
    if args.command in ("geraete", "devices"):
        from .audio import list_devices

        print(list_devices())
        return 0
    if args.command in ("datei", "file"):
        return run_file(args)
    if args.command == "live":
        return run_live(args)
    build_parser().print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
