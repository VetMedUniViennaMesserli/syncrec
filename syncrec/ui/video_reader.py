"""Frame-accurate random access to a recorded video, through the ffmpeg binary.

The recorder already needs ffmpeg, so the viewer uses it too: no extra Python
packages, and anything ffmpeg can encode (libx264, ffv1, mjpeg, ...) can be read.

Reading forward keeps one ffmpeg process streaming raw frames, so stepping and
playing cost one decode per frame. A jump starts a new process with an accurate
seek (ffmpeg decodes from the previous keyframe and discards up to the target).
Stepping backwards is served from a small cache; when it runs dry the reader seeks
a block back and refills it, so a long run of backward steps stays fast.
"""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

CACHE_FRAMES = 40          # per video, full resolution
BACK_BLOCK = 20            # frames decoded ahead of a backward seek


def find_ffprobe(ffmpeg: str) -> Optional[str]:
    folder, name = os.path.split(ffmpeg)
    if folder:
        cand = os.path.join(folder, name.replace("ffmpeg", "ffprobe"))
        if os.path.exists(cand):
            return cand
    return shutil.which("ffprobe")


@dataclass
class FrameInfo:
    """What the recorder's CSV says about one video frame."""
    trigger_id: int = -1
    placeholder: bool = False
    cam_ns: int = 0


def read_frame_info(csv_path: str) -> List[FrameInfo]:
    """Per video frame, in order. Rows for frames dropped before encoding are skipped."""
    out: List[FrameInfo] = []
    if not os.path.exists(csv_path):
        return out
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("written") == "0":
                continue
            try:
                out.append(FrameInfo(trigger_id=int(row.get("trigger_id") or -1),
                                     placeholder=row.get("placeholder") == "1",
                                     cam_ns=int(row.get("cam_timestamp_ns") or 0)))
            except ValueError:
                out.append(FrameInfo())
    return out


class VideoReader:
    def __init__(self, path: str, ffmpeg: str = "ffmpeg", fps_hint: float = 0.0):
        self.path = path
        self.ffmpeg = ffmpeg
        self.name = os.path.splitext(os.path.basename(path))[0].rsplit("_", 1)[-1]
        self.width, self.height, self.fps, self.frame_count = self._probe(fps_hint)
        self.frame_bytes = self.width * self.height * 3
        self.info = read_frame_info(os.path.splitext(path)[0] + ".csv")
        if self.frame_count <= 0 and self.info:
            self.frame_count = len(self.info)
        self._proc: Optional[subprocess.Popen] = None
        self._next = 0                     # index the running process returns next
        self._cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._lock = threading.Lock()

    # ---- probing -------------------------------------------------------------
    def _probe(self, fps_hint: float):
        ffprobe = find_ffprobe(self.ffmpeg)
        if ffprobe:
            r = subprocess.run(
                [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                 "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames",
                 "-of", "json", self.path], capture_output=True, text=True)
            if r.returncode == 0:
                st = (json.loads(r.stdout).get("streams") or [{}])[0]
                fps = self._rate(st.get("avg_frame_rate")) or \
                    self._rate(st.get("r_frame_rate")) or fps_hint or 25.0
                n = int(st.get("nb_frames") or 0)
                if n <= 0:
                    c = subprocess.run(
                        [ffprobe, "-v", "error", "-select_streams", "v:0",
                         "-count_packets", "-show_entries", "stream=nb_read_packets",
                         "-of", "csv=p=0", self.path], capture_output=True, text=True)
                    n = int((c.stdout.strip() or "0").split(",")[0] or 0)
                return int(st["width"]), int(st["height"]), fps, n
        # No ffprobe: read the stream line from ffmpeg's banner.
        r = subprocess.run([self.ffmpeg, "-hide_banner", "-i", self.path],
                           capture_output=True, text=True)
        m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", r.stderr)
        if not m:
            raise RuntimeError(f"cannot read video size of {self.path}")
        f = re.search(r"([\d.]+) fps", r.stderr)
        fps = float(f.group(1)) if f else (fps_hint or 25.0)
        return int(m.group(1)), int(m.group(2)), fps, 0

    @staticmethod
    def _rate(text: Optional[str]) -> float:
        try:
            num, den = (text or "0/0").split("/")
            return float(num) / float(den) if float(den) else 0.0
        except (ValueError, ZeroDivisionError):
            return 0.0

    # ---- decoding ------------------------------------------------------------
    def _start(self, index: int) -> None:
        self._stop()
        cmd = [self.ffmpeg, "-v", "error", "-nostdin"]
        if index > 0:
            # Half a frame early: the first frame at or after this time is `index`,
            # whatever rounding the container's timestamps have.
            cmd += ["-ss", f"{(index - 0.5) / self.fps:.6f}"]
        cmd += ["-i", self.path, "-map", "0:v:0", "-f", "rawvideo",
                "-pix_fmt", "rgb24", "-"]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL,
                                      bufsize=self.frame_bytes * 2)
        self._next = index

    def _stop(self) -> None:
        if self._proc is not None:
            try:
                self._proc.kill()
                self._proc.wait(timeout=2)
            except Exception:
                pass
            self._proc = None

    def _read_one(self) -> Optional[np.ndarray]:
        if self._proc is None or self._proc.stdout is None:
            return None
        buf = bytearray()
        while len(buf) < self.frame_bytes:
            chunk = self._proc.stdout.read(self.frame_bytes - len(buf))
            if not chunk:
                self._stop()
                return None
            buf += chunk
        arr = np.frombuffer(bytes(buf), np.uint8).reshape(self.height, self.width, 3)
        self._remember(self._next, arr)
        self._next += 1
        return arr

    def _remember(self, index: int, arr: np.ndarray) -> None:
        self._cache[index] = arr
        self._cache.move_to_end(index)
        while len(self._cache) > CACHE_FRAMES:
            self._cache.popitem(last=False)

    def frame(self, index: int) -> Optional[np.ndarray]:
        """RGB image of video frame `index`, or None past the end of the video."""
        if index < 0 or (self.frame_count and index >= self.frame_count):
            return None
        with self._lock:
            if index in self._cache:
                self._cache.move_to_end(index)
                return self._cache[index]
            if self._proc is not None and self._next <= index <= self._next + 2 * BACK_BLOCK:
                pass                                   # just read forward to it
            elif self._proc is not None and index < self._next and \
                    index >= self._next - BACK_BLOCK * 3:
                # Stepping backwards past the cache: refill a block that ends here.
                self._start(max(0, index - BACK_BLOCK + 1))
            else:
                self._start(index)
            arr = None
            while self._next <= index:
                arr = self._read_one()
                if arr is None:
                    # The video ended earlier than its header claimed.
                    if self._next and (not self.frame_count or self._next < self.frame_count):
                        self.frame_count = self._next
                    return None
            return self._cache.get(index, arr)

    def frame_info(self, index: int) -> Optional[FrameInfo]:
        return self.info[index] if 0 <= index < len(self.info) else None

    def close(self) -> None:
        with self._lock:
            self._stop()
            self._cache.clear()
