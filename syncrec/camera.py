"""Backend-independent camera interface.

Everything above this module (sync strategies, writers, session, UI) talks only to
`CameraBackend`, so a real Basler camera and a dummy behave identically.
"""
from __future__ import annotations

import enum
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

INVALID_BLOCK_ID = 0xFFFFFFFFFFFFFFFF


class Capability(str, enum.Enum):
    SOFTWARE_TRIGGER = "software_trigger"
    HARDWARE_TRIGGER = "hardware_trigger"
    ACTION_COMMAND = "action_command"        # GigE action commands (needed for PTP sync)
    PTP = "ptp"                              # IEEE 1588 clock sync
    CHUNK_TIMESTAMP = "chunk_timestamp"      # per-frame device timestamp in the payload
    LINE_OUTPUT = "line_output"              # can drive a trigger line for other cameras


@dataclass
class CameraInfo:
    serial: str
    model: str = ""
    device_class: str = ""                   # BaslerGigE / BaslerUsb / BaslerCamEmu / Dummy
    friendly_name: str = ""
    ip_address: str = ""
    user_id: str = ""

    @property
    def is_gige(self) -> bool:
        return "gige" in self.device_class.lower()

    def label(self) -> str:
        return f"{self.serial} ({self.model})" if self.model else self.serial


@dataclass
class Frame:
    """One grabbed image plus everything needed to prove it is in sync."""
    serial: str
    index: int                               # sequential per camera, counted by us
    array: np.ndarray
    host_timestamp_ns: int                   # time.time_ns() at retrieval
    cam_timestamp_ns: int = 0                # device clock; PTP-disciplined when available
    block_id: int = INVALID_BLOCK_ID         # camera frame counter
    trigger_id: int = -1                     # which trigger/action this frame answers
    skipped_before: int = 0                  # frames the camera reports as lost before this one
    pixel_format: str = "Mono8"

    @property
    def is_color(self) -> bool:
        return self.array.ndim == 3 and self.array.shape[2] == 3

    def sync_timestamp_ns(self) -> int:
        """Best available clock for cross-camera comparison."""
        return self.cam_timestamp_ns or self.host_timestamp_ns


class CameraError(RuntimeError):
    pass


class CameraBackend:
    """One physical (or simulated) camera.

    Lifecycle: open() -> configure() -> [sync strategy arms triggers] -> start()
               -> retrieve()* -> stop() -> close()
    """

    info: CameraInfo

    # ---- lifecycle ---------------------------------------------------------
    def open(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def is_open(self) -> bool:
        raise NotImplementedError

    # ---- capability / parameter access ------------------------------------
    def capabilities(self) -> set:
        return set()

    def has(self, cap: Capability) -> bool:
        return cap in self.capabilities()

    def configure(self, cfg, serial_overrides=True) -> None:
        """Apply resolution, pixel format, exposure, gain and frame rate."""
        raise NotImplementedError

    def apply_live(self, cfg) -> None:
        """Apply the settings a camera accepts while grabbing: exposure, gain, fps.

        Used by the live preview, so the effect is visible immediately.
        """

    def output_geometry(self):
        """(width, height, ffmpeg_pix_fmt) of the frames this camera will deliver.

        Asked before grabbing starts, so the writers can be opened up front. It must
        account for any conversion the backend applies (e.g. Bayer -> BGR8).
        """
        raise NotImplementedError

    def get_param(self, name: str, default: Any = None) -> Any:
        return default

    def set_param(self, name: str, value: Any) -> bool:
        return False

    # ---- acquisition -------------------------------------------------------
    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def is_grabbing(self) -> bool:
        raise NotImplementedError

    def retrieve(self, timeout_ms: int) -> Optional[Frame]:
        """Return the next frame, or None on timeout. Raises CameraError on failure."""
        raise NotImplementedError

    def execute_software_trigger(self) -> None:
        raise NotImplementedError

    def wait_for_trigger_ready(self, timeout_ms: int = 1000) -> bool:
        return True

    # ---- sync support ------------------------------------------------------
    def set_trigger(self, mode: str, source: Optional[str] = None,
                    activation: Optional[str] = None) -> None:
        """mode: 'off' (free run) or 'on' with the given trigger source."""
        raise NotImplementedError

    def configure_ptp(self, enable: bool) -> None:
        raise NotImplementedError

    def ptp_status(self) -> Dict[str, Any]:
        """{'enabled', 'state', 'offset_ns'} -- state is e.g. Master/Slave/Listening."""
        return {"enabled": False, "state": "unsupported", "offset_ns": None}

    def configure_action_keys(self, device_key: int, group_key: int, group_mask: int) -> None:
        raise NotImplementedError

    def statistics(self) -> Dict[str, Any]:
        """Backend-reported counters (failed buffers, resends, ...)."""
        return {}

    # ---- helpers -----------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} {getattr(self, 'info', None)}>"


def now_ns() -> int:
    return time.time_ns()
