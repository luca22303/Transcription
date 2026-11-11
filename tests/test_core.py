"""Fast tests that need neither a microphone nor a speech model.

    pip install pytest && pytest
"""

import json
import wave
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest

from lecture_transcriber.audio import SAMPLE_RATE, StreamingResampler
from lecture_transcriber.engine import chunks_of_file
from lecture_transcriber.export import build_docx, default_filename
from lecture_transcriber.output import TranscriptWriter, WavRecorder, fmt_clock, fmt_srt, slugify
from lecture_transcriber.segmenter import Segmenter
from lecture_transcriber.transcriber import Segment, clean_segments


def speech_like(seconds, freq=220.0, amp=0.3, seed=0):
    """Amplitude-modulated tone, loud enough to count as speech for the segmenter."""
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    env = 0.6 + 0.4 * np.sin(2 * np.pi * 3 * t)
    return (amp * env * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(seconds, noise=0.002, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.normal(0, noise, int(seconds * SAMPLE_RATE))).astype(np.float32)


def feed(seg, audio, block=1600):
    chunks = []
    for i in range(0, len(audio), block):
        chunks += seg.push(audio[i : i + block])
    return chunks


# --- segmenter ---------------------------------------------------------------
def test_segmenter_cuts_at_pause():
    audio = np.concatenate([speech_like(8), silence(1.0), speech_like(8), silence(1.0), speech_like(3)])
    seg = Segmenter(min_seconds=6, max_seconds=25)
    chunks = feed(seg, audio) + seg.flush()
    assert len(chunks) >= 2
    # First cut lies inside the first pause (8 s .. 9 s).
    assert 8.0 <= chunks[0].end <= 9.0
    # Nothing is lost and chunks are contiguous.
    assert sum(len(c.audio) for c in chunks) == len(audio)
    for a, b in zip(chunks, chunks[1:]):
        assert abs(a.end - b.start) < 1e-6


def test_segmenter_forces_cut_without_pause():
    audio = speech_like(60)
    seg = Segmenter(min_seconds=6, max_seconds=25)
    chunks = feed(seg, audio) + seg.flush()
    assert all(len(c.audio) <= 25 * SAMPLE_RATE for c in chunks)
    assert sum(len(c.audio) for c in chunks) == len(audio)


def test_segmenter_marks_silence():
    seg = Segmenter(min_seconds=6, max_seconds=25)
    chunks = feed(seg, np.concatenate([speech_like(10), silence(1), silence(30, seed=1)])) + seg.flush()
    assert chunks[0].has_speech
    assert not chunks[-1].has_speech


def test_chunks_of_file_covers_everything():
    audio = np.concatenate([speech_like(7), silence(0.8), speech_like(12)])
    chunks = list(chunks_of_file(audio))
    assert sum(len(c.audio) for c in chunks) == len(audio)


# --- resampler ---------------------------------------------------------------
@pytest.mark.parametrize("src", [48000, 44100, 32000])
def test_resampler_length_and_frequency(src):
    t = np.arange(src * 2) / src
    x = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    rs = StreamingResampler(src)
    out = np.concatenate([rs(x[i : i + 4800]) for i in range(0, len(x), 4800)])
    assert abs(len(out) - 2 * SAMPLE_RATE) <= 3
    spectrum = np.abs(np.fft.rfft(out[: SAMPLE_RATE]))
    peak = np.argmax(spectrum) * SAMPLE_RATE / SAMPLE_RATE  # 1 s window -> bin == Hz
    assert abs(peak - 440) <= 2


# --- hallucination filter -------------------------------------------------------
def raw(text, start=0.0, end=2.0, nsp=0.1, lp=-0.3):
    return SimpleNamespace(start=start, end=end, text=text, no_speech_prob=nsp, avg_logprob=lp)


def test_clean_segments_drops_hallucinations():
    segs = [
        raw(" Heute geht es um Eigenwerte."),
        raw(" Untertitel im Auftrag des ZDF, 2021"),
        raw(" Vielen Dank fürs Zuschauen!"),
        raw(" ja ja ja ja ja ja ja ja ja"),
        raw(" Unsicher", nsp=0.9, lp=-1.5),
        raw(" Außerhalb der Sprache", start=20, end=22),
    ]
    out = clean_segments(segs, offset=100.0, regions=[(0.0, 10.0)])
    assert [s.text for s in out] == ["Heute geht es um Eigenwerte."]
    assert out[0].start == 100.0


def test_clean_segments_accepts_dicts():
    seg = {"start": 1.0, "end": 2.0, "text": " Hallo", "no_speech_prob": 0.0, "avg_logprob": -0.2}
    assert clean_segments([seg])[0].text == "Hallo"


# --- output -------------------------------------------------------------------
def test_formatting():
    assert fmt_clock(3725.4) == "01:02:05"
    assert fmt_srt(61.5) == "00:01:01,500"
    assert slugify("Analysis I: Folgen/Reihen") == "Analysis-I-FolgenReihen"


def test_writer_and_wav(tmp_path):
    w = TranscriptWriter(tmp_path, "Analysis I", header_note="test")
    w.add_paragraph([Segment(1.0, 3.0, "Erster Satz."), Segment(3.0, 5.0, "Zweiter Satz.")])
    w.close()
    md = (tmp_path / "transkript.md").read_text("utf-8")
    assert md.startswith("# Analysis I")
    assert "**[00:00:01]** Erster Satz. Zweiter Satz." in md
    assert "00:00:03,000 --> 00:00:05,000" in (tmp_path / "transkript.srt").read_text("utf-8")

    rec = WavRecorder(tmp_path / "a.wav")
    rec.write(speech_like(1.5))
    # Header is valid even before close() (crash safety).
    with wave.open(str(tmp_path / "a.wav")) as wf:
        assert wf.getnframes() == int(1.5 * SAMPLE_RATE)
    rec.close()


def test_docx_is_valid_zip_with_text():
    data = build_docx("Analysis <I>", "10.11.2026", [{"time": "00:00:01", "text": "a & b"}])
    import io

    z = zipfile.ZipFile(io.BytesIO(data))
    doc = z.read("word/document.xml").decode("utf-8")
    assert "Analysis &lt;I&gt;" in doc and "a &amp; b" in doc
    from xml.dom.minidom import parseString

    parseString(doc)


def test_default_filename():
    info = {"title": "Lineare Algebra: VL 3", "started": "2026-11-10T10:15:00"}
    assert default_filename(info, "docx") == "Lineare Algebra VL 3 2026-11-10.docx"


# --- server security ------------------------------------------------------------
def test_server_rejects_foreign_requests(tmp_path):
    import http.client

    from lecture_transcriber.engine import Engine
    from lecture_transcriber.server import AppServer

    engine = Engine(str(tmp_path))
    server = AppServer(engine, port=0)
    server.start()
    try:
        def req(method, path, headers=None, body=None):
            c = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
            c.request(method, path, body=body, headers=headers or {})
            r = c.getresponse()
            data = r.read()
            c.close()
            return r.status, data

        assert req("GET", "/api/sessions")[0] == 200
        assert req("GET", "/api/sessions", {"Host": "evil.example"})[0] == 403
        assert req("POST", "/api/stop", body="{}")[0] == 403  # missing X-Vorlesung header
        assert req("POST", "/api/stop", {"X-Vorlesung": "1"}, "{}")[0] == 200
        assert req("GET", "/api/sessions/..%2F..%2Fetc")[0] == 404
        status, body = req("GET", "/")
        assert status == 200 and b"Neue Aufnahme" in body
    finally:
        server.close()


def test_archive_roundtrip(tmp_path):
    from lecture_transcriber.engine import Engine

    engine = Engine(str(tmp_path))
    session = engine._new_session("Analysis I", "live")
    w = TranscriptWriter(session, "Analysis I")
    w.add_paragraph([Segment(5.0, 7.0, "Grenzwerte.")])
    w.close()
    engine._finish_info(session, duration=7.0)
    items = engine.list_sessions()
    assert items[0]["title"] == "Analysis I" and items[0]["duration"] == 7.0
    data = engine.load_session(items[0]["id"])
    assert data["paragraphs"] == [{"time": "00:00:05", "text": "Grenzwerte."}]
    assert json.loads((session / "info.json").read_text())["kind"] == "live"


# --- recording process -------------------------------------------------------------
def test_microphone_process_survives_busy_main_process():
    """Audio is captured in its own process, so a main process that blocks the
    GIL (like the speech model does) must not cause gaps in the recording."""
    import threading
    import time

    from lecture_transcriber.audio import MicrophoneSource

    src = MicrophoneSource(fake=True)
    src.start()
    received = []
    reader = threading.Thread(target=lambda: received.extend(src.blocks()))
    reader.start()
    time.sleep(0.5)
    t0 = time.monotonic()
    _ = 7 ** (3 * 10 ** 6)  # big-int math in C: holds the GIL without releasing it
    blocked = time.monotonic() - t0
    time.sleep(0.5)
    src.stop()
    reader.join(10)
    assert not reader.is_alive()
    audio = np.concatenate(received)
    # The fake microphone produces one continuous 440 Hz tone; a gap would break it.
    t = np.arange(len(audio)) / SAMPLE_RATE
    expected = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    assert np.allclose(audio, expected, atol=1e-4)
    assert len(audio) >= (1.0 + blocked - 0.3) * SAMPLE_RATE
    assert src.dropouts == 0
