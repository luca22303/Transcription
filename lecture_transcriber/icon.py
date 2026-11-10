"""Generates the app icon as a PNG (numpy + zlib only).

    python -m lecture_transcriber.icon out.png [size]
"""

from __future__ import annotations

import struct
import sys
import zlib

import numpy as np


def _png(rgba: np.ndarray) -> bytes:
    h, w, _ = rgba.shape
    raw = b"".join(b"\x00" + rgba[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def _rounded_rect_alpha(x, y, cx, cy, hw, hh, r):
    """Anti-aliased coverage of a rounded rectangle (signed distance)."""
    dx = np.maximum(np.abs(x - cx) - (hw - r), 0)
    dy = np.maximum(np.abs(y - cy) - (hh - r), 0)
    d = np.sqrt(dx ** 2 + dy ** 2) - r
    return np.clip(0.5 - d, 0, 1)


def render(size: int = 1024) -> np.ndarray:
    s = size
    y, x = np.mgrid[0:s, 0:s].astype(np.float64) + 0.5
    img = np.zeros((s, s, 4))

    # Background: macOS-style rounded square with a diagonal gradient.
    margin = s * 0.1
    body = _rounded_rect_alpha(x, y, s / 2, s / 2, s / 2 - margin, s / 2 - margin, s * 0.185)
    t = np.clip((x + y) / (2 * s), 0, 1)[..., None]
    top = np.array([0.20, 0.42, 1.00])
    bottom = np.array([0.43, 0.22, 0.86])
    img[..., :3] = top * (1 - t) + bottom * t
    img[..., 3] = body

    # Waveform bars.
    heights = [0.16, 0.30, 0.46, 0.30, 0.52, 0.36, 0.20]
    bar_w = s * 0.052
    gap = s * 0.038
    total = len(heights) * bar_w + (len(heights) - 1) * gap
    start = s / 2 - total / 2 + bar_w / 2
    white = np.zeros((s, s))
    for i, h in enumerate(heights):
        cx = start + i * (bar_w + gap)
        a = _rounded_rect_alpha(x, y, cx, s * 0.54, bar_w / 2, s * h / 2, bar_w / 2)
        white = np.maximum(white, a)
    img[..., :3] = img[..., :3] * (1 - white[..., None]) + white[..., None]

    # Red "recording" dot.
    d = np.sqrt((x - s * 0.765) ** 2 + (y - s * 0.235) ** 2) - s * 0.05
    dot = np.clip(0.5 - d, 0, 1)
    ring = np.clip(0.5 - (np.sqrt((x - s * 0.765) ** 2 + (y - s * 0.235) ** 2) - s * 0.068), 0, 1)
    img[..., :3] = img[..., :3] * (1 - ring[..., None]) + ring[..., None]  # white ring
    red = np.array([1.0, 0.27, 0.23])
    img[..., :3] = img[..., :3] * (1 - dot[..., None]) + red * dot[..., None]
    img[..., 3] = np.maximum(img[..., 3], ring)

    return (np.clip(img, 0, 1) * 255).round().astype(np.uint8)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "icon.png"
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    with open(out, "wb") as fh:
        fh.write(_png(render(size)))


if __name__ == "__main__":
    main()
