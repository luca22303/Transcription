"""Writes transcripts incrementally to disk (Markdown, plain text, SRT).

Every segment is flushed immediately, so nothing is lost if the laptop
crashes or the battery runs out in the middle of a lecture.
"""

from __future__ import annotations

import wave
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .audio import SAMPLE_RATE
from .transcriber import Segment


def fmt_clock(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def fmt_srt(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def slugify(text: str) -> str:
    keep = []
    for ch in text.strip():
        if ch.isalnum():
            keep.append(ch)
        elif ch in " -_":
            keep.append("-")
    slug = "".join(keep).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug[:60]


def new_session_dir(base: str, title: str) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    name = stamp + (f"_{slugify(title)}" if title else "")
    path = Path(base) / name
    i = 2
    while path.exists():
        path = Path(base) / f"{name}_{i}"
        i += 1
    path.mkdir(parents=True)
    return path


class TranscriptWriter:
    def __init__(self, directory: Path, title: str, header_note: str = "", stem: str = "transkript"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.md_path = self.directory / f"{stem}.md"
        self.txt_path = self.directory / f"{stem}.txt"
        self.srt_path = self.directory / f"{stem}.srt"
        self._md = open(self.md_path, "w", encoding="utf-8")
        self._txt = open(self.txt_path, "w", encoding="utf-8")
        self._srt = open(self.srt_path, "w", encoding="utf-8")
        self._srt_index = 0
        self._title = title or "Vorlesung"
        self._note = header_note
        self._header_written = False

    def set_note(self, note: str):
        self._note = note

    def _ensure_header(self):
        if self._header_written:
            return
        self._header_written = True
        self._md.write(f"# {self._title}\n\n")
        self._md.write(f"*{datetime.now().strftime('%d.%m.%Y, %H:%M Uhr')}*")
        if self._note:
            self._md.write(f" · *{self._note}*")
        self._md.write("\n\n")
        self._md.flush()

    def add_paragraph(self, segments: Iterable[Segment]):
        """One paragraph per chunk of speech, prefixed with its start time."""
        segments = list(segments)
        if not segments:
            return
        self._ensure_header()
        text = " ".join(s.text for s in segments).strip()
        self._md.write(f"**[{fmt_clock(segments[0].start)}]** {text}\n\n")
        self._txt.write(text + "\n\n")
        for seg in segments:
            self._srt_index += 1
            self._srt.write(
                f"{self._srt_index}\n{fmt_srt(seg.start)} --> {fmt_srt(seg.end)}\n{seg.text}\n\n"
            )
        for fh in (self._md, self._txt, self._srt):
            fh.flush()

    def add_note(self, text: str):
        self._ensure_header()
        self._md.write(f"> {text}\n\n")
        self._md.flush()

    def close(self):
        if self._md.closed:
            return
        self._ensure_header()
        for fh in (self._md, self._txt, self._srt):
            fh.close()


class WavRecorder:
    """Saves the raw audio (16 kHz mono, 16 bit) as a backup and for re-transcription.

    The WAV header is updated with every write, so the file stays readable
    even if the program is killed.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._wav: Optional[wave.Wave_write] = wave.open(str(self.path), "wb")
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)
        self._wav.setframerate(SAMPLE_RATE)

    def write(self, block: np.ndarray):
        if self._wav is None:
            return
        pcm = np.clip(block, -1.0, 1.0)
        self._wav.writeframes((pcm * 32767).astype("<i2").tobytes())

    def close(self):
        if self._wav is not None:
            self._wav.close()
            self._wav = None
