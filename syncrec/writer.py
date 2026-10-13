"""Writing frames to disk: one ffmpeg process per camera, plus sync metadata.

Frames are piped to ffmpeg as raw video, so the codec is entirely up to
`encoder.output_args` in the config -- libx264, h264_nvenc, ffv1, mjpeg, anything the
installed ffmpeg supports. Alongside each video a CSV records the timing of every
frame, which is what lets you prove afterwards that the cameras really were in sync.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import subprocess
import threading
import sys
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np

from .camera import Frame, INVALID_BLOCK_ID

log = logging.getLogger(__name__)

# numpy dtype/shape -> ffmpeg rawvideo pixel format
PIX_FMTS = {
    ("uint8", 1): "gray",
    ("uint8", 3): "bgr24",
    ("uint16", 1): "gray16le",
    ("uint16", 3): "bgr48le",
}


def ffmpeg_pix_fmt(array: np.ndarray) -> str:
    channels = 1 if array.ndim == 2 else array.shape[2]
    key = (str(array.dtype), channels)
    if key not in PIX_FMTS:
        raise ValueError(f"no ffmpeg pixel format for {array.dtype} with "
                         f"{channels} channel(s)")
    return PIX_FMTS[key]


# GenICam pixel format -> ffmpeg rawvideo pixel format, for the formats we emit
# after the backend's own conversion step.
GENICAM_PIX_FMTS = {
    "Mono8": "gray", "Mono10": "gray", "Mono12": "gray", "Mono16": "gray",
    "RGB8Packed": "rgb24", "RGB8": "rgb24",
    "BGR8Packed": "bgr24", "BGR8": "bgr24",
    "BGRA8Packed": "bgra",
}


def genicam_pix_fmt(name: str) -> str:
    """ffmpeg pixel format for a GenICam pixel format name.

    Bayer formats never reach here as Bayer: the pylon backend debayers them to BGR8
    before the frame is handed on, and deep mono formats are converted down to Mono8.
    """
    if name in GENICAM_PIX_FMTS:
        return GENICAM_PIX_FMTS[name]
    if name.startswith("Bayer"):
        return "bgr24"
    if name.startswith("Mono"):
        return "gray"
    raise ValueError(f"unsupported pixel format {name!r}")


def resolve_ffmpeg(candidate: str) -> str:
    """Find the ffmpeg binary, falling back to the usual Windows install path."""
    found = shutil.which(candidate)
    if found:
        return found
    for path in (r"C:\ffmpeg\bin\ffmpeg.exe", r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"):
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"ffmpeg not found (tried {candidate!r} and the usual install paths). "
        f"Set encoder.ffmpeg in the config to its full path.")


class FfmpegWriter:
    """Pipes raw frames of one camera into one ffmpeg process."""

    def __init__(self, path: str, width: int, height: int, fps: float,
                 pix_fmt: str, cfg_encoder, segment_frames: int = 0):
        # With segment_frames, `path` is an ffmpeg segment template
        # (".../segment_%04d/name.mp4") and ffmpeg starts a new file every
        # segment_frames frames, inside the same encoder process.
        self.path = path
        self.segment_frames = int(segment_frames or 0)
        self.width = width
        self.height = height
        self.fps = fps
        self.pix_fmt = pix_fmt
        self._cfg = cfg_encoder
        self._proc: Optional[subprocess.Popen] = None
        self.frames_written = 0
        self.bytes_written = 0
        self._frame_nbytes = 0

    def command(self) -> List[str]:
        exe = resolve_ffmpeg(self._cfg.ffmpeg)
        cmd = [exe, "-hide_banner", "-loglevel", self._cfg.loglevel, "-y",
               "-f", "rawvideo", "-pix_fmt", self.pix_fmt,
               "-s", f"{self.width}x{self.height}", "-r", f"{self.fps:g}"]
        cmd += list(self._cfg.input_args)
        cmd += ["-i", "pipe:0", "-an"]
        cmd += list(self._cfg.output_args)
        if self.segment_frames:
            cmd += self._segment_args()
        cmd += [self.path]
        return cmd

    _SEGMENT_FORMATS = {".mp4": "mp4", ".mov": "mov", ".mkv": "matroska",
                        ".avi": "avi", ".mxf": "mxf"}

    def _segment_args(self) -> List[str]:
        """Split into files of exactly segment_frames frames, without restarting.

        A segment can only start on a key frame, so one is forced on every
        segment_frames-th frame (counted, not timed -- exact at any frame rate).
        The segment muxer then cuts at that key frame: the frame timestamps are
        n / fps, so the cut time is exactly segment_frames / fps, and the half-frame
        tolerance stops a nearby ordinary key frame from being taken instead.
        """
        n = self.segment_frames
        args = ["-force_key_frames", f"expr:eq(mod(n,{n}),0)"]
        # Not every encoder honours forced key frames (SVT-AV1 ignores them), so
        # also make the regular key frame interval divide the segment length: then
        # every cut point is a key frame either way.
        if "-g" not in self._cfg.output_args:
            target = max(1, int(round(self.fps * 10)))
            gop = max(d for d in range(1, min(n, target) + 1) if n % d == 0)
            args += ["-g", str(gop)]
        if any("nvenc" in a for a in self._cfg.output_args):
            args += ["-forced-idr", "1"]       # NVENC: forced frames must be IDR
        ext = os.path.splitext(self.path)[1].lower()
        args += ["-f", "segment",
                 "-segment_format", self._SEGMENT_FORMATS.get(ext, ext.lstrip(".")),
                 "-segment_time", f"{n / self.fps:.9f}",
                 "-segment_time_delta", f"{0.5 / self.fps:.9f}",
                 "-reset_timestamps", "1"]
        return args

    def open(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        cmd = self.command()
        log.info("ffmpeg: %s", " ".join(cmd))
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        # SVT-AV1 prints a banner through its own logger, ignoring ffmpeg's log
        # level; keep it to errors so real problems are not buried.
        env = dict(os.environ)
        env.setdefault("SVT_LOG", "1")
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                      stdout=subprocess.DEVNULL,
                                      stderr=subprocess.PIPE,
                                      creationflags=creationflags, env=env)
        # Read ffmpeg's messages continuously. If nobody reads them, the pipe fills
        # (64 KB) after enough warnings and ffmpeg blocks -- in a recording that runs
        # for days that eventually happens, and the encoder silently stalls.
        self._stderr_lines: deque = deque(maxlen=40)
        self._stderr_thread = threading.Thread(target=self._read_stderr,
                                               args=(self._proc,), daemon=True,
                                               name=f"ffmpeg-log-{os.path.basename(self.path)}")
        self._stderr_thread.start()

    def _read_stderr(self, proc) -> None:
        try:
            for raw in iter(proc.stderr.readline, b""):
                line = raw.decode("utf-8", "replace").rstrip()
                if line:
                    self._stderr_lines.append(line)
                    log.debug("ffmpeg %s: %s", os.path.basename(self.path), line)
        except Exception:
            pass

    def write(self, array: np.ndarray) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError(f"writer for {self.path} is not open")
        if self._proc.poll() is not None:
            raise RuntimeError(f"ffmpeg exited early for {self.path}: "
                               f"{self._stderr_tail()}")
        data = array.tobytes()
        if not self._frame_nbytes:
            self._frame_nbytes = len(data)
        elif len(data) != self._frame_nbytes:
            raise RuntimeError(f"{self.path}: frame size changed mid-recording "
                               f"({self._frame_nbytes} -> {len(data)} bytes)")
        self._proc.stdin.write(data)
        self.frames_written += 1
        self.bytes_written += len(data)

    def close(self, timeout: float = 60.0) -> Optional[str]:
        """Flush and wait for ffmpeg. Returns an error string, or None on success."""
        if self._proc is None:
            return None
        proc, self._proc = self._proc, None
        try:
            if proc.stdin:
                try:
                    proc.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            return f"ffmpeg for {self.path} did not exit within {timeout}s; killed"
        thread = getattr(self, "_stderr_thread", None)
        if thread is not None:
            thread.join(timeout=5)
        err = self._stderr_text()
        if proc.returncode not in (0, None):
            return f"ffmpeg for {self.path} exited {proc.returncode}: {err}"
        return None

    def _stderr_tail(self) -> str:
        thread = getattr(self, "_stderr_thread", None)
        if thread is not None:
            thread.join(timeout=2)
        return self._stderr_text()

    def _stderr_text(self) -> str:
        return "\n".join(getattr(self, "_stderr_lines", ()))[-800:]


class MetadataWriter:
    """Per-frame timing CSV: the evidence that the cameras were in sync."""

    COLUMNS = ["frame_index", "block_id", "trigger_id", "cam_timestamp_ns",
               "host_timestamp_ns", "skipped_before", "queued_frames", "written",
               "placeholder"]

    def __init__(self, path: str):
        self.path = path
        self._fh = None
        self._writer: Optional[csv.writer] = None
        self.rows = 0
        # Rows come from the write thread and (for dropped frames) the grab thread.
        self._lock = threading.Lock()

    def open(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self._fh = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(self.COLUMNS)

    def write(self, frame: Frame, queued: int, written: bool) -> None:
        block = "" if frame.block_id == INVALID_BLOCK_ID else frame.block_id
        with self._lock:
            if self._writer is None:
                return
            self._writer.writerow([frame.index, block, frame.trigger_id,
                                   frame.cam_timestamp_ns, frame.host_timestamp_ns,
                                   frame.skipped_before, queued, int(written), 0])
            self.rows += 1

    def write_placeholder(self, trigger_id: int) -> None:
        """A video frame that repeats the previous image because trigger_id was missed."""
        with self._lock:
            if self._writer is None:
                return
            self._writer.writerow([-1, "", trigger_id, 0, 0, 0, 0, 1, 1])
            self.rows += 1

    def switch(self, path: str) -> None:
        """Continue in a new file (next segment), with its own header."""
        with self._lock:
            if self._fh is not None:
                self._fh.close()
            self.path = path
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self._fh = open(path, "w", newline="", encoding="utf-8")
            self._writer = csv.writer(self._fh)
            self._writer.writerow(self.COLUMNS)

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.flush()
                self._fh.close()
                self._fh = None
                self._writer = None


def find_ffprobe(ffmpeg: str) -> Optional[str]:
    exe = resolve_ffmpeg(ffmpeg)
    folder, name = os.path.split(exe)
    if folder:
        cand = os.path.join(folder, name.replace("ffmpeg", "ffprobe"))
        if os.path.exists(cand):
            return cand
    return shutil.which("ffprobe")


def count_video_frames(ffprobe: str, path: str) -> int:
    """Frames in a finished video file (-1 if it cannot be read)."""
    for extra in ([], ["-count_packets"]):
        entry = "nb_read_packets" if extra else "nb_frames"
        try:
            r = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0"] + extra +
                               ["-show_entries", f"stream={entry}", "-of", "csv=p=0", path],
                               capture_output=True, text=True, timeout=600)
            value = r.stdout.strip().split(",")[0]
            if value.isdigit() and int(value) > 0:
                return int(value)
        except (OSError, subprocess.SubprocessError):
            pass
    return -1


def write_manifest(path: str, payload: Dict[str, Any]) -> None:
    """Session-level manifest: config, cameras, counts, sync report."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
        fh.write("\n")
    os.replace(tmp, path)
