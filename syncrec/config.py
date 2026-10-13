"""Configuration schema for the multi-camera recorder.

A whole recording setup is one JSON file (see config.example.json). settings.json is
loaded automatically when no other file is given (see DEFAULT_CONFIG_PATH); without it
the defaults below apply. Everything the operator can tune -- resolution, frame rate,
sync strategy, ffmpeg codec, per-camera overrides -- lives there, so the recording
machine is configured by copying one file.
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional, Union, get_args, get_origin, get_type_hints

SYNC_MODES = ("auto", "ptp_action", "hardware_trigger", "software_trigger", "freerun")
BACKENDS = ("auto", "pylon", "dummy")



def _settings_path() -> str:
    """Where settings.json lives ("Save as default" in the GUI writes it).

    The SYNCREC_SETTINGS environment variable, if set: the Linux installer's
    `syncrec` command sets it to a per-user file (~/.config/syncrec/settings.json),
    because the program folder is not writable. Otherwise next to the syncrec
    package, wherever the program is started from (not part of the repository).
    """
    path = os.environ.get("SYNCREC_SETTINGS")
    if path:
        return os.path.abspath(os.path.expanduser(path))
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "settings.json")


DEFAULT_CONFIG_PATH = _settings_path()


@dataclass
class Roi:
    width: int = 2048
    height: int = 1536
    offset_x: int = 0
    offset_y: int = 0


@dataclass
class SessionConfig:
    output_dir: str = "recordings"
    # Recording name: names the folder and (via {clip}) the files. {date} and {time}
    # are filled in when the recording starts; blank means "{date}_{time}".
    clip_name: str = "{date}_{time}"
    filename_pattern: str = "{clip}_{serial}.mp4"
    duration_s: float = 0.0                  # 0 = record until stopped
    # When a camera misses a frame, write a copy of its previous frame in that slot
    # so video frame n is trigger n in every camera's file and the videos stay
    # aligned. Needs a sync mode that can tell which trigger a frame belongs to
    # (ptp_action); the CSV marks every filled frame as a placeholder.
    fill_missing_frames: bool = True
    # Split long recordings into segments of this many minutes (0 = one file).
    # Each segment goes into its own folder, segment_0000/, segment_0001/, ...,
    # with the same file names and per-frame CSVs as an unsplit recording. The
    # split happens inside the running encoder at an exact frame count, so no
    # frame is lost or duplicated and every camera splits at the same trigger.
    segment_minutes: float = 0.0
    # Stop the recording cleanly when the disk gets this full (GB free), rather
    # than letting the encoders fail mid-file. 0 = no check.
    min_free_gb: float = 5.0


@dataclass
class AcquisitionConfig:
    backend: str = "auto"                    # auto | pylon | dummy
    fps: float = 25.0
    exposure_us: float = 5000.0
    gain_db: float = 0.0
    pixel_format: str = "Mono8"
    roi: Roi = field(default_factory=Roi)
    # Reduce resolution WITHOUT cropping: the camera combines binning x binning
    # sensor pixels into one, so the whole field of view arrives with 1/4 (2x2) of
    # the data. roi width/height are then in binned pixels (2048x1536 sensor,
    # binning 2 -> full view is 1024x768). Most Basler colour models cannot bin
    # (it would mix the Bayer colours); then decimation (skipping pixels) is used if
    # the camera has it, else the recording refuses to start rather than crop.
    binning: int = 1
    packet_size: int = 8192                  # GigE only; ignored elsewhere
    # GigE only. Pause between the packets of one camera, so several cameras can share
    # one link. Give it in microseconds: the camera's own unit ("ticks") changes
    # length when PTP is switched on, so a raw tick value silently becomes 8x shorter
    # on ace classic cameras. inter_packet_delay (ticks of 8 ns) is the old form and
    # is used only when inter_packet_delay_us is not set.
    inter_packet_delay_us: Optional[float] = None
    inter_packet_delay: int = 0
    queue_size: int = 120                    # frames buffered per camera before dropping
    grab_timeout_ms: int = 5000


@dataclass
class PtpConfig:
    # The first lock after the cameras power up can take several minutes when the
    # target offset is tight. Later recordings reuse the lock (keep_enabled).
    sync_timeout_s: float = 300.0
    # Worst clock offset of any slave to the master before recording may start; this
    # is the trigger skew between cameras.
    max_offset_ns: int = 1000
    # The offsets must stay within max_offset_ns for this long, not just touch it once.
    settle_s: float = 5.0
    # Leave PTP running on the cameras after a recording. They then stay locked, and
    # the next recording starts as soon as the lock is re-verified (settle_s).
    keep_enabled: bool = True
    # How far ahead of its exposure each action command is sent. An upper limit:
    # it is reduced automatically so that no more commands are pending at once than
    # the cameras can queue (ace classic: one). Otherwise the camera rejects the
    # extra commands and only takes every 2nd or 3rd trigger.
    action_lead_time_ms: float = 100.0
    device_key: int = 0x12345678
    group_key: int = 0x12345678
    group_mask: int = 0x1
    # "auto": the directed broadcast of the cameras' subnet (e.g. 169.254.255.255).
    broadcast_address: str = "auto"


@dataclass
class HardwareTriggerConfig:
    trigger_source: str = "Line1"
    trigger_activation: str = "RisingEdge"
    # Optional: let one camera generate the trigger for all the others.
    master_serial: Optional[str] = None
    master_line: str = "Line2"
    master_line_source: str = "ExposureActive"


@dataclass
class SyncConfig:
    mode: str = "auto"
    ptp: PtpConfig = field(default_factory=PtpConfig)
    hardware: HardwareTriggerConfig = field(default_factory=HardwareTriggerConfig)
    max_spread_ms: float = 2.0               # warn if cameras drift further apart than this


@dataclass
class EncoderConfig:
    """Fully configurable ffmpeg invocation.

    The recorder pipes raw frames to ffmpeg's stdin; `output_args` is passed through
    verbatim, so any codec ffmpeg supports can be used (libx264, h264_nvenc, ffv1,
    mjpeg, ...).
    """
    ffmpeg: str = "ffmpeg"
    # Codec and encoder hardware. When codec is set, the ffmpeg arguments are
    # generated (see syncrec/encoders.py) and output_args/input_args are ignored;
    # codec "" or "custom" uses output_args verbatim.
    #   codec:    h264 | h265 | av1 | mjpeg | ffv1 | custom
    #   hardware: auto | cpu | nvidia | intel | amd | vaapi
    #   quality:  lower = better and bigger; None = codec default (H.264 20,
    #             H.265 24, AV1 32)
    codec: str = ""
    hardware: str = "auto"
    quality: Optional[int] = None
    vaapi_device: str = "/dev/dri/renderD128"
    output_args: List[str] = field(default_factory=lambda: [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
    ])
    input_args: List[str] = field(default_factory=list)
    loglevel: str = "error"
    write_metadata_csv: bool = True


@dataclass
class CameraOverride:
    enabled: bool = True
    name: str = ""                           # friendly label shown in the UI
    exposure_us: Optional[float] = None
    gain_db: Optional[float] = None
    roi: Optional[Roi] = None
    rotate_180: bool = False                 # for a camera mounted upside down


@dataclass
class DummyConfig:
    count: int = 8
    width: int = 1024
    height: int = 768
    jitter_ms: float = 0.3                   # per-camera timestamp jitter in freerun
    drop_rate: float = 0.0                   # fraction of frames the camera "loses"
    serial_prefix: str = "DUMMY"


@dataclass
class RecorderConfig:
    session: SessionConfig = field(default_factory=SessionConfig)
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    cameras: Dict[str, CameraOverride] = field(default_factory=dict)
    dummy: DummyConfig = field(default_factory=DummyConfig)

    # ---- (de)serialisation -------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        folder = os.path.dirname(os.path.abspath(path))
        os.makedirs(folder, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
            fh.write("\n")
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "RecorderConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecorderConfig":
        cfg = _build(cls, data or {})
        cfg.validate()
        return cfg

    def override_for(self, serial: str) -> CameraOverride:
        return self.cameras.get(serial, CameraOverride())

    def exposure_for(self, serial: str) -> float:
        ov = self.override_for(serial)
        return self.acquisition.exposure_us if ov.exposure_us is None else ov.exposure_us

    def gain_for(self, serial: str) -> float:
        ov = self.override_for(serial)
        return self.acquisition.gain_db if ov.gain_db is None else ov.gain_db

    def roi_for(self, serial: str) -> Roi:
        ov = self.override_for(serial)
        return ov.roi if ov.roi is not None else self.acquisition.roi

    def validate(self) -> None:
        if self.sync.mode not in SYNC_MODES:
            raise ValueError(f"sync.mode must be one of {SYNC_MODES}, got {self.sync.mode!r}")
        if self.acquisition.backend not in BACKENDS:
            raise ValueError(f"acquisition.backend must be one of {BACKENDS}")
        if self.acquisition.fps <= 0:
            raise ValueError("acquisition.fps must be > 0")
        if self.acquisition.queue_size < 2:
            raise ValueError("acquisition.queue_size must be >= 2")
        r = self.acquisition.roi
        if r.width <= 0 or r.height <= 0:
            raise ValueError("acquisition.roi width/height must be > 0")
        if not 1 <= int(self.acquisition.binning) <= 4:
            raise ValueError("acquisition.binning must be 1, 2, 3 or 4")
        from .encoders import CODECS, HARDWARE
        codec = (self.encoder.codec or "").lower()
        if codec not in ("", "custom") and codec not in CODECS:
            raise ValueError(f"encoder.codec must be one of {list(CODECS)} or 'custom'")
        if (self.encoder.hardware or "auto").lower() not in HARDWARE:
            raise ValueError(f"encoder.hardware must be one of {list(HARDWARE)}")


def _build(tp, data):
    """Recursively construct nested dataclasses from plain dicts, ignoring extras."""
    hints = get_type_hints(tp)
    kwargs = {}
    for f in fields(tp):
        if f.name not in data:
            continue
        kwargs[f.name] = _coerce(hints[f.name], data[f.name])
    return tp(**kwargs)


def _coerce(hint, value):
    """Turn a JSON value into whatever `hint` asks for (dataclass / dict / scalar)."""
    if value is None:
        return None
    origin = get_origin(hint)
    if origin is Union:                                   # Optional[X]
        inner = [a for a in get_args(hint) if a is not type(None)]
        return _coerce(inner[0], value) if len(inner) == 1 else copy.deepcopy(value)
    if origin is dict:
        _, vt = get_args(hint)
        return {str(k): _coerce(vt, v) for k, v in value.items()}
    if is_dataclass(hint) and isinstance(value, dict):
        return _build(hint, value)
    return copy.deepcopy(value)
