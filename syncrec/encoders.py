"""Video codec and encoder hardware selection.

Instead of writing ffmpeg arguments by hand, the config can name a codec (H.264,
H.265, AV1, ...) and where to encode it (CPU or a GPU vendor). This module turns
that into ffmpeg arguments and checks that the combination really works on this PC.

"Works" is tested, not assumed: ffmpeg builds list GPU encoders whether or not a
suitable GPU and driver are present, and GPUs differ in what they support (AV1
encoding needs a recent generation). The test also runs one encoder per camera at
the same time, because consumer GPUs limit how many encodes run in parallel.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

CODECS: Dict[str, str] = {
    "h264": "H.264",
    "h265": "H.265 / HEVC",
    "av1": "AV1",
    "mjpeg": "MJPEG (large files)",
    "ffv1": "FFV1 (lossless, very large)",
}

HARDWARE: Dict[str, str] = {
    "auto": "Auto (GPU if available, else CPU)",
    "cpu": "CPU",
    "nvidia": "NVIDIA GPU (NVENC)",
    "intel": "Intel GPU (Quick Sync)",
    "amd": "AMD GPU (AMF)",
    "vaapi": "Intel/AMD GPU on Linux (VAAPI)",
}

GPU_ORDER = ["nvidia", "intel", "amd", "vaapi"]

# Lower = better quality, bigger files. One scale per codec family, roughly matched
# so the defaults look alike: x264/NVENC CQ ~20, HEVC ~24, AV1 ~32.
DEFAULT_QUALITY = {"h264": 20, "h265": 24, "av1": 32, "mjpeg": 3, "ffv1": 0}

_ENCODER = {
    ("h264", "cpu"): "libx264", ("h265", "cpu"): "libx265",
    ("av1", "cpu"): "libsvtav1", ("mjpeg", "cpu"): "mjpeg", ("ffv1", "cpu"): "ffv1",
    ("h264", "nvidia"): "h264_nvenc", ("h265", "nvidia"): "hevc_nvenc",
    ("av1", "nvidia"): "av1_nvenc",
    ("h264", "intel"): "h264_qsv", ("h265", "intel"): "hevc_qsv", ("av1", "intel"): "av1_qsv",
    ("h264", "amd"): "h264_amf", ("h265", "amd"): "hevc_amf", ("av1", "amd"): "av1_amf",
    ("h264", "vaapi"): "h264_vaapi", ("h265", "vaapi"): "hevc_vaapi",
    ("av1", "vaapi"): "av1_vaapi",
}


def encoder_name(codec: str, hw: str) -> Optional[str]:
    return _ENCODER.get((codec, hw))


def build_args(codec: str, hw: str, quality: Optional[int] = None,
               vaapi_device: str = "/dev/dri/renderD128",
               container: str = ".mp4") -> Optional[Tuple[List[str], List[str]]]:
    """(input_args, output_args) for ffmpeg, or None if the combination does not exist."""
    enc = encoder_name(codec, hw)
    if enc is None:
        return None
    q = str(DEFAULT_QUALITY.get(codec, 20) if quality is None else int(quality))
    inp: List[str] = []
    if codec == "mjpeg":
        out = ["-c:v", "mjpeg", "-q:v", q, "-pix_fmt", "yuvj420p"]
    elif codec == "ffv1":
        out = ["-c:v", "ffv1", "-level", "3", "-g", "1", "-slices", "4"]
    elif hw == "cpu":
        if codec == "h264":
            out = ["-c:v", enc, "-preset", "veryfast", "-crf", q, "-pix_fmt", "yuv420p"]
        elif codec == "h265":
            out = ["-c:v", enc, "-preset", "veryfast", "-crf", q, "-pix_fmt", "yuv420p",
                   "-x265-params", "log-level=error"]
        else:   # av1, SVT-AV1: presets 10-12 are its fast end
            out = ["-c:v", enc, "-preset", "10", "-crf", q, "-pix_fmt", "yuv420p"]
    elif hw == "nvidia":
        out = ["-c:v", enc, "-preset", "p4", "-rc", "vbr", "-cq", q, "-b:v", "0",
               "-pix_fmt", "yuv420p"]
    elif hw == "intel":
        out = ["-c:v", enc, "-preset", "veryfast", "-global_quality", q,
               "-pix_fmt", "nv12"]
    elif hw == "amd":
        out = (["-c:v", enc, "-quality", "speed", "-rc", "cqp", "-qp_i", q, "-qp_p", q]
               + (["-qp_b", q] if codec != "av1" else []) + ["-pix_fmt", "nv12"])
    else:       # vaapi: frames are uploaded to the GPU by a filter
        inp = ["-vaapi_device", vaapi_device]
        out = ["-vf", "format=nv12,hwupload", "-c:v", enc, "-rc_mode", "CQP", "-qp", q]
    # Lets QuickTime and most players recognise HEVC in MP4/MOV.
    if codec == "h265" and container.lower() in (".mp4", ".mov"):
        out += ["-tag:v", "hvc1"]
    return inp, out


def _listed_encoders(ffmpeg: str) -> str:
    try:
        r = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True,
                           text=True, timeout=20)
        return r.stdout
    except (OSError, subprocess.SubprocessError):
        return ""


_test_cache: Dict[Tuple[str, str, str, int], Tuple[bool, str]] = {}


def test(ffmpeg: str, codec: str, hw: str, sessions: int = 1,
         vaapi_device: str = "/dev/dri/renderD128") -> Tuple[bool, str]:
    """Encode a short test clip with `sessions` encoders at once. (ok, message)."""
    key = (ffmpeg, codec, hw, sessions)
    if key in _test_cache:
        return _test_cache[key]
    built = build_args(codec, hw, vaapi_device=vaapi_device, container=".mkv")
    enc = encoder_name(codec, hw)
    if built is None or enc is None:
        result = (False, f"{CODECS.get(codec, codec)} is not available on "
                         f"{HARDWARE.get(hw, hw)}")
        _test_cache[key] = result
        return result
    if f" {enc} " not in _listed_encoders(ffmpeg):
        result = (False, f"this ffmpeg has no {enc} encoder")
        _test_cache[key] = result
        return result
    inp, out = built
    # Real time (-re) for one second, so the parallel encoders actually overlap.
    realtime = ["-re"] if hw != "cpu" else []
    cmd = ([ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin"] + inp + realtime +
           ["-f", "lavfi", "-i", "testsrc2=size=1024x768:rate=25", "-frames:v", "25"]
           + out + ["-f", "null", "-"])
    errors: List[str] = []

    def run() -> None:
        try:
            env = dict(os.environ)
            env.setdefault("SVT_LOG", "1")
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=env)
            if r.returncode != 0:
                # The first line names the cause ("Cannot load libcuda.so.1", "No VA
                # display found", ...); later lines are generic consequences.
                first = (r.stderr.strip().splitlines() or ["failed"])[0]
                errors.append(re.sub(r"^\[[^\]]*@ 0x[0-9a-f]+\]\s*", "", first))
        except (OSError, subprocess.SubprocessError) as exc:
            errors.append(str(exc))

    threads = [threading.Thread(target=run) for _ in range(max(1, sessions))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        n_ok = max(1, sessions) - len(errors)
        msg = errors[0]
        if sessions > 1 and n_ok > 0:
            msg = (f"only {n_ok} of {sessions} parallel encodes worked (the GPU or its "
                   f"driver limits parallel sessions): {msg}")
        result = (False, msg)
    else:
        result = (True, f"{enc} works" + (f" with {sessions} parallel encodes"
                                          if sessions > 1 else ""))
    _test_cache[key] = result
    return result


def resolve(enc_cfg, sessions: int, container: str = ".mp4"
            ) -> Tuple[List[str], List[str], str]:
    """Final (input_args, output_args, description) for a recording.

    Raises RuntimeError if an explicitly requested encoder does not work, rather
    than silently recording with something else.
    """
    from .writer import resolve_ffmpeg
    ffmpeg = resolve_ffmpeg(enc_cfg.ffmpeg)
    codec = (getattr(enc_cfg, "codec", "") or "").lower()
    if codec in ("", "custom"):
        return list(enc_cfg.input_args), list(enc_cfg.output_args), "custom ffmpeg arguments"
    if codec not in CODECS:
        raise RuntimeError(f"unknown codec {codec!r}; use one of {', '.join(CODECS)}")
    hw = (getattr(enc_cfg, "hardware", "auto") or "auto").lower()
    quality = getattr(enc_cfg, "quality", None)
    device = getattr(enc_cfg, "vaapi_device", "/dev/dri/renderD128")
    if codec in ("mjpeg", "ffv1"):
        hw = "cpu"                                   # CPU-only formats
    candidates = GPU_ORDER + ["cpu"] if hw == "auto" else [hw]
    reasons = []
    for cand in candidates:
        if encoder_name(codec, cand) is None:
            continue
        if cand == "cpu":
            ok, msg = test(ffmpeg, codec, "cpu", 1, device)
        else:
            ok, msg = test(ffmpeg, codec, cand, sessions, device)
        if ok:
            inp, out = build_args(codec, cand, quality, device, container)
            desc = f"{CODECS[codec]} on {HARDWARE[cand]} ({encoder_name(codec, cand)})"
            if hw == "auto" and reasons:
                desc += " - not available: " + "; ".join(reasons)
            return list(enc_cfg.input_args) + inp, out, desc
        reasons.append(f"{HARDWARE[cand]}: {msg}")
        log.info("encoder %s / %s not usable: %s", codec, cand, msg)
    raise RuntimeError(f"no working encoder for {CODECS[codec]}: " + "; ".join(reasons))


def survey(ffmpeg: str, sessions: int = 1) -> List[Tuple[str, str, bool, str]]:
    """Test every codec on every hardware option: (codec, hw, ok, message)."""
    rows = []
    for codec in CODECS:
        for hw in ["cpu"] + GPU_ORDER:
            if encoder_name(codec, hw) is None:
                continue
            ok, msg = test(ffmpeg, codec, hw, 1 if hw == "cpu" else sessions)
            rows.append((codec, hw, ok, msg))
    return rows


def container_of(pattern: str) -> str:
    return os.path.splitext(pattern)[1] or ".mp4"
