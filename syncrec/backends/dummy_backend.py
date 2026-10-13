"""Dummy cameras: a full CameraBackend with no hardware.

Unlike the pylon camera emulator (which is also supported, and is the better test of
the real code path), these fake the things the emulator cannot: device timestamps,
PTP convergence, action commands, frame drops and per-camera clock drift. That makes
them the only way to exercise the sync logic itself without cameras on the bench.
"""
from __future__ import annotations

import queue
import random
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

from ..camera import (Capability, CameraBackend, CameraError, CameraInfo, Frame,
                      now_ns)
from .scene import VirtualRig


class DummyClock:
    """A device clock that is deliberately imperfect until PTP disciplines it."""

    def __init__(self, offset_ns: int, drift_ppm: float):
        self.offset_ns = offset_ns
        self.drift_ppm = drift_ppm
        self._t0 = time.time_ns()
        self.ptp_locked = False

    def now_ns(self) -> int:
        host = time.time_ns()
        if self.ptp_locked:
            # Residual error of a converged PTP slave: tens of nanoseconds.
            return host + random.randint(-80, 80)
        elapsed = host - self._t0
        return host + self.offset_ns + int(elapsed * self.drift_ppm * 1e-6)


class DummyCamera(CameraBackend):
    def __init__(self, serial: str, index: int, rig: VirtualRig, cfg_dummy,
                 t_zero: Optional[float] = None):
        self.info = CameraInfo(serial=serial, model="DummyCam",
                               device_class="Dummy", friendly_name=f"Dummy ({serial})",
                               ip_address=f"10.0.0.{10 + index}")
        self._index = index
        self._rig = rig
        self._dummy_cfg = cfg_dummy
        self._t_zero = t_zero if t_zero is not None else time.time()

        self._open = False
        self._grabbing = False
        self._fps = 25.0
        self._exposure_us = 5000.0
        self._gain_db = 0.0
        self._pixel_format = "Mono8"
        self._rotate_180 = False

        self._trigger_mode = "off"
        self._trigger_source = "Software"
        self._ptp_enabled = False
        self._ptp_lock_at: Optional[float] = None
        self._action_keys = (0, 0, 0)

        self._clock = DummyClock(offset_ns=random.randint(-5_000_000, 5_000_000),
                                 drift_ppm=random.uniform(-40, 40))
        self._q: "queue.Queue[Frame]" = queue.Queue(maxsize=256)
        # Triggers are answered on a worker thread. A real camera returns from its
        # trigger immediately and exposes on its own time; doing the rendering inline
        # would make the host's trigger loop pay for it and distort the timing.
        self._triggers: "queue.Queue[tuple]" = queue.Queue(maxsize=256)
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._block = 0
        self._index_counter = 0
        self._skipped = 0
        self._lock = threading.Lock()

    # ---- lifecycle ---------------------------------------------------------
    def open(self) -> None:
        self._open = True

    def close(self) -> None:
        if self._grabbing:
            self.stop()
        self._open = False

    def is_open(self) -> bool:
        return self._open

    def capabilities(self) -> set:
        return {Capability.SOFTWARE_TRIGGER, Capability.HARDWARE_TRIGGER,
                Capability.ACTION_COMMAND, Capability.PTP,
                Capability.CHUNK_TIMESTAMP, Capability.LINE_OUTPUT}

    # ---- parameters --------------------------------------------------------
    def configure(self, cfg, serial_overrides=True) -> None:
        acq = cfg.acquisition
        self._fps = acq.fps
        self._pixel_format = acq.pixel_format
        s = self.info.serial
        self._exposure_us = cfg.exposure_for(s) if serial_overrides else acq.exposure_us
        self._gain_db = cfg.gain_for(s) if serial_overrides else acq.gain_db
        roi = cfg.roi_for(s) if serial_overrides else acq.roi
        self._rotate_180 = cfg.override_for(s).rotate_180
        self._rig = VirtualRig(max(self._rig.n_cams, 1), roi.width, roi.height)
        self._index_counter = 0          # like a real camera: counting restarts

    def apply_live(self, cfg) -> None:
        s = self.info.serial
        self._exposure_us = cfg.exposure_for(s)
        self._gain_db = cfg.gain_for(s)
        self._fps = cfg.acquisition.fps

    def output_geometry(self):
        color = self._pixel_format.startswith("RGB") or self._pixel_format.startswith("BGR")
        return self._rig.width, self._rig.height, ("bgr24" if color else "gray")

    def get_param(self, name: str, default: Any = None) -> Any:
        return {"Width": self._rig.width, "Height": self._rig.height,
                "PixelFormat": self._pixel_format, "ExposureTime": self._exposure_us,
                "Gain": self._gain_db, "AcquisitionFrameRate": self._fps,
                "TriggerMode": "On" if self._trigger_mode == "on" else "Off",
                "TriggerSource": self._trigger_source}.get(name, default)

    def set_param(self, name: str, value: Any) -> bool:
        if name == "ExposureTime":
            self._exposure_us = float(value)
        elif name == "Gain":
            self._gain_db = float(value)
        elif name == "AcquisitionFrameRate":
            self._fps = float(value)
        else:
            return False
        return True

    # ---- acquisition -------------------------------------------------------
    def start(self) -> None:
        if not self._open:
            raise CameraError(f"{self.info.serial}: start() before open()")
        self._stop_evt.clear()
        # Frames left over from a previous run (e.g. the live preview) must not end
        # up in the next one; a real camera's StartGrabbing starts empty too.
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self._grabbing = True
        loop = self._freerun_loop if self._trigger_mode == "off" else self._trigger_loop
        self._thread = threading.Thread(target=loop, name=f"dummy-{self.info.serial}",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._grabbing = False
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def is_grabbing(self) -> bool:
        return self._grabbing

    def retrieve(self, timeout_ms: int) -> Optional[Frame]:
        try:
            return self._q.get(timeout=max(0.001, timeout_ms / 1000.0))
        except queue.Empty:
            return None

    def execute_software_trigger(self) -> None:
        if not self._grabbing:
            raise CameraError(f"{self.info.serial}: software trigger while not grabbing")
        self._enqueue_trigger(-1, self._clock.now_ns())

    def wait_for_trigger_ready(self, timeout_ms: int = 1000) -> bool:
        return self._grabbing

    def fire_external_trigger(self, trigger_id: int, trigger_ns: int) -> None:
        """Called by the sync layer to simulate a hardware/action-command trigger."""
        if self._grabbing:
            self._enqueue_trigger(trigger_id, trigger_ns)

    def _enqueue_trigger(self, trigger_id: int, trigger_ns: int) -> None:
        try:
            self._triggers.put_nowait((trigger_id, trigger_ns))
        except queue.Full:
            # A real camera ignores triggers it is not ready for, and reports the
            # missed frame; mirror that rather than blocking the caller.
            self._skipped += 1

    # ---- sync --------------------------------------------------------------
    def set_trigger(self, mode: str, source: Optional[str] = None,
                    activation: Optional[str] = None) -> None:
        self._trigger_mode = "on" if mode == "on" else "off"
        if source:
            self._trigger_source = source

    def configure_ptp(self, enable: bool) -> None:
        # Like the real camera: re-enabling keeps an existing lock.
        if bool(enable) == self._ptp_enabled:
            return
        self._ptp_enabled = enable
        # Converge after a plausible delay rather than instantly.
        self._ptp_lock_at = time.time() + random.uniform(1.5, 4.0) if enable else None
        self._clock.ptp_locked = False

    def ptp_status(self) -> Dict[str, Any]:
        if not self._ptp_enabled:
            return {"enabled": False, "state": "Off", "offset_ns": None}
        locked = self._ptp_lock_at is not None and time.time() >= self._ptp_lock_at
        self._clock.ptp_locked = locked
        state = ("Master" if self._index == 0 else "Slave") if locked else "Listening"
        offset = random.randint(-120, 120) if locked else random.randint(-900_000, 900_000)
        return {"enabled": True, "state": state, "offset_ns": offset}

    def configure_action_keys(self, device_key: int, group_key: int,
                              group_mask: int) -> None:
        self._action_keys = (device_key, group_key, group_mask)

    def statistics(self) -> Dict[str, Any]:
        return {"queued": self._q.qsize(), "skipped_total": self._skipped,
                "clock_drift_ppm": round(self._clock.drift_ppm, 1)}

    # ---- internals ---------------------------------------------------------
    def _freerun_loop(self) -> None:
        # Free-running cameras start at slightly different phases -- exactly the
        # de-sync that the triggered modes exist to remove.
        next_t = time.time() + random.uniform(0, 1.0 / max(1e-6, self._fps))
        while not self._stop_evt.is_set():
            wait = next_t - time.time()
            if wait > 0 and self._stop_evt.wait(wait):
                break
            jitter = random.gauss(0, self._dummy_cfg.jitter_ms / 1000.0)
            self._emit(trigger_ns=self._clock.now_ns() + int(jitter * 1e9))
            next_t += 1.0 / max(1e-6, self._fps)       # fps may change while running

    def _trigger_loop(self) -> None:
        while not self._stop_evt.is_set():
            try:
                trigger_id, trigger_ns = self._triggers.get(timeout=0.1)
            except queue.Empty:
                continue
            self._emit(trigger_ns=trigger_ns, trigger_id=trigger_id)

    def _emit(self, trigger_ns: int, trigger_id: int = -1) -> None:
        with self._lock:
            if self._dummy_cfg.drop_rate > 0 and random.random() < self._dummy_cfg.drop_rate:
                self._block += 1
                self._skipped += 1
                return
            t_s = (trigger_ns / 1e9) - self._t_zero
            color = self._pixel_format.startswith("RGB") or self._pixel_format.startswith("BGR")
            img = self._rig.render(self._index, t_s, self._block, self.info.serial,
                                   color=color)
            # Exposure and gain brighten the image as they would on a real sensor
            # (5000 us, 0 dB = as rendered), so their effect shows in the preview.
            scale = self._exposure_us / 5000.0 * 10 ** (self._gain_db / 20.0)
            if abs(scale - 1.0) > 1e-3:
                img = np.clip(img * scale, 0, 255).astype(np.uint8)
            if self._rotate_180:
                img = np.ascontiguousarray(np.rot90(img, 2))
            frame = Frame(serial=self.info.serial, index=self._index_counter, array=img,
                          host_timestamp_ns=now_ns(), cam_timestamp_ns=trigger_ns,
                          block_id=self._block, trigger_id=trigger_id,
                          skipped_before=self._skipped, pixel_format=self._pixel_format)
            self._block += 1
            self._index_counter += 1
            self._skipped = 0
        try:
            self._q.put_nowait(frame)
        except queue.Full:
            pass


def discover_dummy(cfg) -> List[DummyCamera]:
    """Create `cfg.dummy.count` dummy cameras sharing one virtual room."""
    d = cfg.dummy
    roi = cfg.acquisition.roi
    rig = VirtualRig(max(1, d.count), roi.width, roi.height)
    t_zero = time.time()
    return [DummyCamera(f"{d.serial_prefix}{i:02d}", i, rig, d, t_zero=t_zero)
            for i in range(d.count)]
