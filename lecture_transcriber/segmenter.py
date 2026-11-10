"""Splits a continuous audio stream into chunks at natural speech pauses.

Whisper works on windows of up to 30 s. To keep latency low and avoid cutting
words in half, we cut the stream at short pauses once a chunk has reached
`min_seconds`, and force a cut at the quietest spot once it reaches
`max_seconds`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List

import numpy as np

from .audio import SAMPLE_RATE

FRAME = int(SAMPLE_RATE * 0.03)  # 30 ms analysis frames


@dataclass
class Chunk:
    audio: np.ndarray
    start: float  # seconds since recording start
    has_speech: bool

    @property
    def end(self) -> float:
        return self.start + len(self.audio) / SAMPLE_RATE


class Segmenter:
    def __init__(
        self,
        min_seconds: float = 6.0,
        max_seconds: float = 25.0,
        silence_seconds: float = 0.5,
    ):
        self.min_samples = int(min_seconds * SAMPLE_RATE)
        self.max_samples = int(max_seconds * SAMPLE_RATE)
        self.silence_frames = max(1, int(silence_seconds * SAMPLE_RATE / FRAME))
        self._buf = np.zeros(0, dtype=np.float32)
        self._offset = 0  # absolute sample index of _buf[0]
        # Rolling history of frame energies (~60 s) to estimate the noise floor.
        self._history: deque = deque(maxlen=2000)

    # -- energy helpers ---------------------------------------------------
    @staticmethod
    def _frame_rms(audio: np.ndarray) -> np.ndarray:
        n = len(audio) // FRAME
        if n == 0:
            return np.zeros(0)
        frames = audio[: n * FRAME].reshape(n, FRAME)
        return np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)

    def _threshold(self) -> float:
        if len(self._history) < 50:
            return 0.01
        hist = np.fromiter(self._history, dtype=np.float64)
        noise_floor = np.percentile(hist, 15)
        speech_level = np.percentile(hist, 90)
        # Pause = clearly closer to the noise floor than to typical speech.
        thr = noise_floor + 0.25 * (speech_level - noise_floor)
        return float(max(thr, 0.002))

    # -- public API -------------------------------------------------------
    def push(self, block: np.ndarray) -> List[Chunk]:
        self._history.extend(self._frame_rms(block).tolist())
        self._buf = np.concatenate([self._buf, block])
        chunks: List[Chunk] = []
        while True:
            cut = self._find_cut()
            if cut is None:
                break
            chunks.append(self._emit(cut))
        return chunks

    def flush(self) -> List[Chunk]:
        if len(self._buf) < SAMPLE_RATE // 2:
            return []
        return [self._emit(len(self._buf))]

    # -- internals --------------------------------------------------------
    def _find_cut(self):
        if len(self._buf) < self.min_samples:
            return None
        rms = self._frame_rms(self._buf)
        thr = self._threshold()
        quiet = rms < thr
        first = self.min_samples // FRAME
        last = min(len(rms), self.max_samples // FRAME)

        # Latest sufficiently long pause after min_seconds -> cut in its middle.
        run = 0
        best = None
        for i in range(first, last):
            run = run + 1 if quiet[i] else 0
            if run >= self.silence_frames:
                best = i - run // 2
        if best is not None and len(self._buf) < self.max_samples:
            # Only cut if speech resumed or the pause is ongoing; either way it's
            # a safe boundary. Prefer earlier emission for low latency.
            return best * FRAME
        if len(self._buf) >= self.max_samples:
            if best is not None:
                return best * FRAME
            # No real pause: cut at the quietest 300 ms in the second half.
            win = max(1, int(0.3 * SAMPLE_RATE / FRAME))
            lo = max(first, len(rms) // 2)
            hi = last
            if hi - lo <= win:
                return self.max_samples
            smoothed = np.convolve(rms[lo:hi], np.ones(win) / win, mode="valid")
            return (lo + int(np.argmin(smoothed)) + win // 2) * FRAME
        return None

    def _emit(self, cut: int) -> Chunk:
        audio = self._buf[:cut]
        rms = self._frame_rms(audio)
        has_speech = bool(len(rms)) and int(np.sum(rms > self._threshold())) >= 5
        chunk = Chunk(audio=audio, start=self._offset / SAMPLE_RATE, has_speech=has_speech)
        self._buf = self._buf[cut:]
        self._offset += cut
        return chunk
