"""Speech recognition backends with filters for typical Whisper hallucinations.

Two backends are supported:

* ``mlx``            – mlx-whisper, runs on the GPU of Apple-Silicon Macs (M1–M4).
                       Fast enough to run ``large-v3-turbo`` live.
* ``faster-whisper`` – CTranslate2, runs on CPU everywhere and on NVIDIA GPUs.

``auto`` picks MLX on Apple Silicon (if installed), otherwise faster-whisper.
"""

from __future__ import annotations

import platform
import re
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .audio import SAMPLE_RATE

# Whisper tends to "hear" these phrases in silence or noise (artefacts from
# subtitled videos in its training data). Segments consisting only of such a
# phrase are dropped.
HALLUCINATIONS = [
    r"untertitel(ung)? (im auftrag )?(des|der|von) .*",
    r"untertitel von .*",
    r"copyright .*",
    r"vielen dank (fürs|für's|für das|für ihre?) zu(schauen|hören|sehen)",
    r"danke fürs zuschauen",
    r"amara\.org.*",
    r"thanks? (you )?for watching.*",
    r"subtitles? by .*",
    r"please subscribe.*",
    r"\[?musik\]?",
    r"\[?music\]?",
    r"\*.*\*",
    r"\.+",
    r"…+",
]
_HALLU_RE = re.compile(r"^\s*(" + "|".join(HALLUCINATIONS) + r")[\s.!?…]*$", re.IGNORECASE)

MLX_REPOS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "turbo": "mlx-community/whisper-large-v3-turbo",
}


@dataclass
class Segment:
    start: float
    end: float
    text: str


def is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def mlx_available() -> bool:
    try:
        import mlx_whisper  # noqa: F401
    except Exception:
        return False
    return True


def resolve_backend(backend: str) -> str:
    if backend == "auto":
        return "mlx" if is_apple_silicon() and mlx_available() else "faster-whisper"
    if backend == "mlx" and not mlx_available():
        raise SystemExit(
            "Das MLX-Backend ist nicht installiert. Auf einem Mac mit Apple Silicon:\n"
            "  pip install mlx-whisper"
        )
    return backend


def _cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def default_model(backend: str) -> str:
    if backend == "mlx" or _cuda_available():
        return "large-v3-turbo"
    return "small"


# ---------------------------------------------------------------------------
# Voice activity detection (Silero VAD, ships with faster-whisper)
# ---------------------------------------------------------------------------
def speech_regions(audio: np.ndarray) -> List[Tuple[float, float]]:
    """Return speech regions in seconds. Used to skip silence and to drop
    segments that Whisper invented during pauses."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    stamps = get_speech_timestamps(
        audio, VadOptions(min_silence_duration_ms=500, speech_pad_ms=300)
    )
    return [(s["start"] / SAMPLE_RATE, s["end"] / SAMPLE_RATE) for s in stamps]


def _overlaps(start: float, end: float, regions: Sequence[Tuple[float, float]]) -> bool:
    for a, b in regions:
        if start < b and end > a:
            return True
    return False


def _get(seg, key):
    return seg[key] if isinstance(seg, dict) else getattr(seg, key)


def clean_segments(
    raw_segments,
    offset: float = 0.0,
    regions: Optional[Sequence[Tuple[float, float]]] = None,
    on_segment: Optional[Callable[[float], None]] = None,
) -> List[Segment]:
    out: List[Segment] = []
    for seg in raw_segments:
        start, end = float(_get(seg, "start")), float(_get(seg, "end"))
        if on_segment:
            on_segment(end)
        text = _get(seg, "text").strip()
        if not text:
            continue
        # Classic Whisper heuristic: probably no speech and low confidence.
        if _get(seg, "no_speech_prob") > 0.6 and _get(seg, "avg_logprob") < -1.0:
            continue
        if _HALLU_RE.match(text):
            continue
        # Endless repetition loops ("ja ja ja ja ja ...")
        words = text.split()
        if len(words) >= 8 and len({w.lower().strip(".,!?") for w in words}) <= 2:
            continue
        if regions is not None and not _overlaps(start, end, regions):
            continue
        out.append(Segment(start=start + offset, end=end + offset, text=text))
    return out


# ---------------------------------------------------------------------------
class Transcriber:
    def __init__(
        self,
        model: Optional[str] = None,
        language: Optional[str] = "de",
        backend: str = "auto",
        device: str = "auto",
        compute_type: str = "auto",
        beam_size: int = 5,
        topic_prompt: str = "",
    ):
        self.backend = resolve_backend(backend)
        self.model_name = model or default_model(self.backend)
        self.language = None if language in (None, "", "auto") else language
        self.beam_size = beam_size
        self.topic_prompt = topic_prompt.strip()
        self._context = ""

        if self.backend == "mlx":
            from concurrent.futures import ThreadPoolExecutor

            self.mlx_repo = MLX_REPOS.get(self.model_name, self.model_name)
            self.device_desc = "Apple GPU (MLX)"
            # MLX binds GPU streams to threads: load and use the model on one
            # dedicated thread, no matter which thread calls us.
            self._mlx_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
            self._mlx_thread.submit(self._warmup_mlx).result()
        else:
            from faster_whisper import WhisperModel

            if device == "auto":
                device = "cuda" if _cuda_available() else "cpu"
            if compute_type == "auto":
                compute_type = "float16" if device == "cuda" else "int8"
            self.device_desc = f"{device}, {compute_type}"
            self.model = WhisperModel(self.model_name, device=device, compute_type=compute_type)

    @property
    def description(self) -> str:
        return f"{self.model_name} via {self.backend} ({self.device_desc})"

    # -- prompt handling --------------------------------------------------
    def reset(self, topic_prompt: str = ""):
        """Start a new lecture: forget the previous context."""
        self.topic_prompt = (topic_prompt or "").strip()
        self._context = ""

    def _prompt(self) -> Optional[str]:
        prompt = " ".join(p for p in (self.topic_prompt, self._context) if p).strip()
        return prompt or None

    def _remember(self, segments: List[Segment]):
        # Keep the tail of what was said as context for the next chunk
        # (improves continuity and the spelling of names / technical terms).
        if segments:
            text = " ".join(s.text for s in segments)
            self._context = (self._context + " " + text).strip()[-300:]

    # -- backend calls ----------------------------------------------------
    def _warmup_mlx(self):
        import mlx_whisper

        # Downloads the model on first use and compiles kernels.
        mlx_whisper.transcribe(
            np.zeros(SAMPLE_RATE, dtype=np.float32),
            path_or_hf_repo=self.mlx_repo,
            language=self.language,
            verbose=None,
        )

    def _run_mlx(self, audio: np.ndarray, prompt: Optional[str]):
        import mlx_whisper

        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.mlx_repo,
            language=self.language,
            initial_prompt=prompt,
            condition_on_previous_text=False,
            verbose=None,
        )
        return result["segments"]

    def _run(self, audio: np.ndarray, prompt: Optional[str]):
        if self.backend == "mlx":
            return self._mlx_thread.submit(self._run_mlx, audio, prompt).result()
        segments, _info = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            initial_prompt=prompt,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        return segments

    # -- public API -------------------------------------------------------
    def transcribe_chunk(self, audio: np.ndarray, offset: float) -> List[Segment]:
        """Transcribe one chunk of live audio. Timestamps are absolute (offset added)."""
        regions = speech_regions(audio)
        if not regions:
            return []
        result = clean_segments(self._run(audio, self._prompt()), offset, regions)
        self._remember(result)
        return result
