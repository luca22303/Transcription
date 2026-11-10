"""Audio sources: live microphone (or loopback device) and audio files.

All sources yield mono float32 blocks at SAMPLE_RATE (16 kHz), which is
what Whisper expects.
"""

from __future__ import annotations

import queue
import time
from typing import Iterator, Optional

import numpy as np

SAMPLE_RATE = 16000
BLOCK_SECONDS = 0.1


def decode_audio(path: str) -> np.ndarray:
    """Decode any audio/video file (WAV, MP3, M4A, MP4, ...) to 16 kHz mono float32.

    Uses PyAV (bundled FFmpeg, installed with faster-whisper). Written here
    instead of using faster_whisper.decode_audio because that one breaks with
    newer PyAV releases.
    """
    import av

    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    pieces = []
    with av.open(str(path), mode="r") as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise SystemExit(f"Keine Tonspur gefunden in: {path}")
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                pieces.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):  # flush
            pieces.append(out.to_ndarray().reshape(-1))
    if not pieces:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(pieces).astype(np.float32) / 32768.0


def _import_sounddevice():
    try:
        import sounddevice as sd
    except OSError as exc:  # PortAudio missing (typically Linux)
        raise SystemExit(
            "PortAudio wurde nicht gefunden. Unter Linux z.B. installieren mit:\n"
            "  sudo apt install libportaudio2\n"
            f"(Originalfehler: {exc})"
        )
    return sd


def input_devices() -> list:
    """Input devices as data for the UI: [{id, name, default}]."""
    sd = _import_sounddevice()
    default_in = sd.default.device[0] if sd.default.device else None
    out = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            out.append({"id": str(idx), "name": dev["name"], "default": idx == default_in})
    return out


def list_devices() -> str:
    sd = _import_sounddevice()
    lines = []
    default_in = sd.default.device[0] if sd.default.device else None
    hostapis = sd.query_hostapis()
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] < 1:
            continue
        marker = "*" if idx == default_in else " "
        api = hostapis[dev["hostapi"]]["name"]
        lines.append(
            f"{marker} [{idx:>2}] {dev['name']}  "
            f"({api}, {int(dev['default_samplerate'])} Hz, {dev['max_input_channels']} Kanäle)"
        )
    if not lines:
        return "Keine Eingabegeräte gefunden."
    return "Eingabegeräte (* = Standard):\n" + "\n".join(lines)


def _resolve_device(sd, device: Optional[str]):
    """Accept a device index or a (case-insensitive) substring of its name."""
    if device is None or device == "":
        return None
    if device.isdigit():
        return int(device)
    matches = [
        idx
        for idx, dev in enumerate(sd.query_devices())
        if dev["max_input_channels"] > 0 and device.lower() in dev["name"].lower()
    ]
    if not matches:
        raise SystemExit(
            f"Kein Eingabegerät mit '{device}' im Namen gefunden. "
            "Verfügbare Geräte: `vorlesung devices`"
        )
    return matches[0]


class StreamingResampler:
    """Small dependency-free streaming resampler (box low-pass + linear interp).

    Good enough for speech recognition. Only used when the audio device
    refuses to record at 16 kHz directly.
    """

    def __init__(self, src_rate: int, dst_rate: int = SAMPLE_RATE):
        self.step = src_rate / dst_rate
        self.kernel = max(1, int(round(self.step)))
        self._tail = np.zeros(0, dtype=np.float32)  # unfiltered history for the filter
        self._buf = np.zeros(0, dtype=np.float32)  # filtered samples not yet consumed
        self._pos = 0.0  # fractional read position inside _buf

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.step == 1:
            return x
        if self.kernel > 1:
            joined = np.concatenate([self._tail, x])
            filtered = np.convolve(joined, np.ones(self.kernel) / self.kernel, mode="valid")
            self._tail = joined[-(self.kernel - 1):]
        else:
            filtered = x
        self._buf = np.concatenate([self._buf, filtered.astype(np.float32)])
        n_out = int((len(self._buf) - 1 - self._pos) // self.step) + 1
        if n_out <= 0:
            return np.zeros(0, dtype=np.float32)
        positions = self._pos + np.arange(n_out) * self.step
        out = np.interp(positions, np.arange(len(self._buf)), self._buf).astype(np.float32)
        next_pos = self._pos + n_out * self.step
        drop = int(next_pos) - 1 if int(next_pos) >= 1 else 0
        self._buf = self._buf[drop:]
        self._pos = next_pos - drop
        return out


class MicrophoneSource:
    """Records continuously from an input device. Runs until stop() is called."""

    def __init__(self, device: Optional[str] = None):
        self.sd = _import_sounddevice()
        self.device = _resolve_device(self.sd, device)
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stream = None
        self._resampler: Optional[StreamingResampler] = None
        self._channels = 1
        self._running = False
        self.overflows = 0

    @property
    def name(self) -> str:
        info = self.sd.query_devices(self.device, "input")
        return info["name"]

    def _callback(self, indata, frames, time_info, status):
        if status and status.input_overflow:
            self.overflows += 1
        self._queue.put(indata.copy())

    def _open(self, samplerate: int, channels: int):
        stream = self.sd.InputStream(
            samplerate=samplerate,
            channels=channels,
            dtype="float32",
            device=self.device,
            blocksize=int(samplerate * BLOCK_SECONDS),
            callback=self._callback,
        )
        stream.start()
        return stream

    def start(self):
        info = self.sd.query_devices(self.device, "input")
        native_rate = int(info["default_samplerate"])
        max_ch = int(info["max_input_channels"])
        attempts = [(SAMPLE_RATE, 1), (native_rate, 1), (native_rate, min(2, max_ch))]
        last_err = None
        for rate, ch in attempts:
            try:
                self._stream = self._open(rate, ch)
            except Exception as exc:  # PortAudioError and friends
                last_err = exc
                continue
            self._channels = ch
            self._resampler = StreamingResampler(rate) if rate != SAMPLE_RATE else None
            self._running = True
            return
        raise SystemExit(f"Audiogerät konnte nicht geöffnet werden: {last_err}")

    def stop(self):
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def blocks(self) -> Iterator[np.ndarray]:
        while self._running or not self._queue.empty():
            try:
                data = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            mono = data.mean(axis=1) if data.ndim == 2 else data
            if self._resampler is not None:
                mono = self._resampler(mono)
            if len(mono):
                yield mono.astype(np.float32, copy=False)


class FileSource:
    """Plays an audio/video file as if it were a live input (for testing/demo)."""

    def __init__(self, path: str, speed: float = 1.0):
        self.path = path
        self.audio = decode_audio(path)
        self.speed = speed
        self._running = False

    @property
    def name(self) -> str:
        return f"Datei {self.path}"

    def start(self):
        self._running = True

    def stop(self):
        self._running = False

    def blocks(self) -> Iterator[np.ndarray]:
        block = int(SAMPLE_RATE * BLOCK_SECONDS)
        t0 = time.monotonic()
        for i in range(0, len(self.audio), block):
            if not self._running:
                break
            if self.speed > 0:
                due = t0 + (i / SAMPLE_RATE) / self.speed
                delay = due - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            yield self.audio[i : i + block]
