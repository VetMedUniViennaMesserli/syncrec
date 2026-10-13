"""A tiny synthetic world used by the dummy cameras.

The point is not pretty pictures: the dummy cameras project the *same* moving object
(four markers on a rigid body) through different known camera poses, so a dummy
recording can be checked against known 3D positions end to end. Each frame also gets its frame number and
timestamp burned in, which makes de-sync visible by eye.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

KPT_NAMES = ["marker_a", "marker_b", "marker_c", "marker_d"]

# 5x7 bitmap font, enough for the burn-in overlay (digits, ':', '.', '-', ' ').
_GLYPHS = {
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11111", "00010", "00100", "00010", "00001", "10001", "01110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
    ":": ("00000", "00100", "00100", "00000", "00100", "00100", "00000"),
    ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    "#": ("01010", "11111", "01010", "01010", "11111", "01010", "00000"),
    " ": ("00000",) * 7,
}
for _c, _rows in list(_GLYPHS.items()):
    _GLYPHS[_c] = np.array([[int(ch) for ch in r] for r in _rows], np.uint8)


_GLYPH_CACHE = {}


def _glyph(ch: str, scale: int) -> np.ndarray:
    key = (ch, scale)
    block = _GLYPH_CACHE.get(key)
    if block is None:
        g = _GLYPHS.get(ch.upper(), _GLYPHS[" "])
        block = np.kron(g, np.ones((scale, scale), np.uint8)).astype(bool)
        _GLYPH_CACHE[key] = block
    return block


def draw_text(img: np.ndarray, text: str, x: int, y: int, scale: int = 3,
              value: int = 255) -> None:
    """Blit `text` at (x, y) using the built-in bitmap font (no OpenCV needed)."""
    cx = x
    for ch in text:
        block = _glyph(ch, scale)
        h, w = block.shape
        if y + h > img.shape[0] or cx + w > img.shape[1]:
            break
        img[y:y + h, cx:cx + w][block] = value
        cx += w + scale


def _disc(img: np.ndarray, cx: float, cy: float, r: float, value) -> None:
    h, w = img.shape[:2]
    x0, x1 = max(0, int(cx - r)), min(w, int(cx + r) + 1)
    y0, y1 = max(0, int(cy - r)), min(h, int(cy + r) + 1)
    if x1 <= x0 or y1 <= y0:
        return
    ys, xs = np.ogrid[y0:y1, x0:x1]
    mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r
    img[y0:y1, x0:x1][mask] = value


def _segment(img: np.ndarray, p0, p1, thickness: float, value) -> None:
    """Fill the capsule between p0 and p1 in one pass.

    Stamping overlapping discs along the line costs ~50 small numpy calls per limb;
    computing point-to-segment distance over the bounding box costs one.
    """
    h, w = img.shape[:2]
    x0 = max(0, int(min(p0[0], p1[0]) - thickness) - 1)
    x1 = min(w, int(max(p0[0], p1[0]) + thickness) + 2)
    y0 = max(0, int(min(p0[1], p1[1]) - thickness) - 1)
    y1 = min(h, int(max(p0[1], p1[1]) + thickness) + 2)
    if x1 <= x0 or y1 <= y0:
        return
    ys, xs = np.ogrid[y0:y1, x0:x1]
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    len2 = dx * dx + dy * dy
    if len2 < 1e-9:
        _disc(img, p0[0], p0[1], thickness, value)
        return
    t = np.clip(((xs - p0[0]) * dx + (ys - p0[1]) * dy) / len2, 0.0, 1.0)
    ex = xs - (p0[0] + t * dx)
    ey = ys - (p0[1] + t * dy)
    img[y0:y1, x0:x1][ex * ex + ey * ey <= thickness * thickness] = value


class VirtualRig:
    """`n` cameras on a circle around a room, all looking at its centre.

    Units are millimetres, matching the real calibration (world = mm).
    """

    def __init__(self, n_cams: int, width: int, height: int,
                 radius_mm: float = 3500.0, height_mm: float = 2400.0):
        self.n_cams = n_cams
        self.width = width
        self.height = height
        f = 0.9 * max(width, height)
        self.K = np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]])
        self.poses: List[Tuple[np.ndarray, np.ndarray]] = []
        for i in range(n_cams):
            a = 2 * math.pi * i / max(1, n_cams)
            C = np.array([radius_mm * math.cos(a), radius_mm * math.sin(a), height_mm])
            self.poses.append(self._look_at(C, np.zeros(3)))
        self._backgrounds: Dict[int, np.ndarray] = {}

    def _background(self, cam: int) -> np.ndarray:
        """Static, camera-specific backdrop, built once and then just copied."""
        bg = self._backgrounds.get(cam)
        if bg is None:
            w, h = self.width, self.height
            grad = np.linspace(28, 86, w, dtype=np.uint8)[None, :].repeat(h, 0)
            bg = np.roll(grad, (cam * 137) % w, axis=1)
            bg[::max(32, h // 12)] = 60
            self._backgrounds[cam] = bg
        return bg

    @staticmethod
    def _look_at(C: np.ndarray, target: np.ndarray):
        fwd = target - C
        fwd = fwd / np.linalg.norm(fwd)
        up = np.array([0.0, 0.0, -1.0])
        right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        true_up = np.cross(fwd, right)
        R = np.stack([right, true_up, fwd])          # world -> camera
        t = -R @ C
        return R, t

    def object_points(self, t_s: float) -> np.ndarray:
        """(4, 3) world positions of the four markers (KPT_NAMES) at time t."""
        cx = 1400.0 * math.sin(0.35 * t_s)
        cy = 1100.0 * math.sin(0.23 * t_s + 1.1)
        heading = math.atan2(1100 * 0.23 * math.cos(0.23 * t_s + 1.1),
                             1400 * 0.35 * math.cos(0.35 * t_s))
        fx, fy = math.cos(heading), math.sin(heading)
        bob = 30.0 * math.sin(6.0 * t_s)
        a = np.array([cx + 540 * fx, cy + 540 * fy, 430 + bob])
        b = np.array([cx + 380 * fx, cy + 380 * fy, 520 + bob])
        c = np.array([cx - 380 * fx, cy - 380 * fy, 500 + bob])
        swing = 0.9 * math.sin(11.0 * t_s)
        sx, sy = math.cos(heading + math.pi + swing), math.sin(heading + math.pi + swing)
        d = c + np.array([260 * sx, 260 * sy, 180.0])
        return np.stack([a, b, c, d])

    def project(self, cam: int, pts_w: np.ndarray) -> np.ndarray:
        R, t = self.poses[cam % self.n_cams]
        pc = pts_w @ R.T + t
        z = np.clip(pc[:, 2], 1e-6, None)
        uv = (pc[:, :2] / z[:, None]) @ self.K[:2, :2].T + self.K[:2, 2]
        uv[pc[:, 2] <= 0] = np.nan
        return uv

    def render(self, cam: int, t_s: float, frame_no: int, serial: str,
               color: bool = False) -> np.ndarray:
        h, w = self.height, self.width
        bg = self._background(cam % self.n_cams)
        if color:
            img = np.repeat(bg[:, :, None], 3, axis=2)
        else:
            img = bg.copy()
        uv = self.project(cam, self.object_points(t_s))
        if not np.isnan(uv).any():
            size = 0.016 * max(w, h)
            for p0, p1 in zip(uv[:-1], uv[1:]):
                _segment(img, p0, p1, size * 0.25, 170)
            for p in uv:
                _disc(img, p[0], p[1], size * 0.8, 250)
        draw_text(img, f"#{frame_no:06d}", 8, 8, scale=max(1, w // 340), value=255)
        draw_text(img, f"{t_s:09.3f}", 8, 8 + 9 * max(1, w // 340), scale=max(1, w // 340),
                  value=255)
        draw_text(img, serial[-6:], w - 7 * 6 * max(1, w // 340) - 8, 8,
                  scale=max(1, w // 340), value=200)
        return img
