"""Live preview: show what the cameras see before (and between) recordings.

The cameras free-run at the configured frame rate with the configured exposure and
image size, so the picture and the network load match what a recording will look
like -- but they are NOT synchronised; that is set up when a recording starts.
Nothing is written to disk.

start() and stop() block (configuring many GigE cameras takes a moment), so the UI
calls them from a worker thread.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from .camera import CameraBackend
from .session import make_thumbnail
from .sync import choose_mode

log = logging.getLogger(__name__)

# Thumbnails per second per camera; more only costs CPU, the UI polls at 5 Hz.
PREVIEW_RATE_HZ = 10.0


class LivePreview:
    def __init__(self, cfg, cameras: Sequence[CameraBackend]):
        self.cfg = cfg
        self.cameras: List[CameraBackend] = list(cameras)
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._threads: List[threading.Thread] = []
        self._started: List[CameraBackend] = []
        self._images: Dict[str, np.ndarray] = {}
        self._stats: Dict[str, Dict[str, object]] = {}
        self._data_lock = threading.Lock()
        self.running = False

    # ---- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            if self.running:
                return
            self._stop_evt.clear()
            # PTP takes a while to lock the first time after power-up. Switching it
            # on now lets it converge while people set up, so Record starts quickly.
            # (The camera only rewrites the setting on an actual change.)
            ptp = choose_mode(self.cfg, self.cameras) == "ptp_action" if self.cameras \
                else False
            for cam in self.cameras:
                serial = cam.info.serial
                self._stats[serial] = {"fps": 0.0, "grabbed": 0, "error": ""}
                try:
                    if not cam.is_open():
                        cam.open()
                    if ptp:
                        try:
                            cam.configure_ptp(True)
                        except Exception as exc:
                            log.debug("%s: PTP not enabled for preview: %s", serial, exc)
                    cam.configure(self.cfg)
                    cam.set_trigger("off")
                    cam.start()
                except Exception as exc:
                    log.error("%s: live preview failed: %s", serial, exc)
                    self._stats[serial]["error"] = str(exc)
                    try:
                        cam.stop()
                    except Exception:
                        pass
                    continue
                self._started.append(cam)
                th = threading.Thread(target=self._grab_loop, args=(cam,),
                                      name=f"preview-{serial}", daemon=True)
                th.start()
                self._threads.append(th)
            self.running = True
            if self._started:
                log.info("live preview running on %d cameras (not synchronised)",
                         len(self._started))

    def stop(self) -> None:
        with self._lock:
            if not self.running:
                return
            self._stop_evt.set()
            for th in self._threads:
                th.join(timeout=3.0)
            self._threads.clear()
            for cam in self._started:
                try:
                    cam.stop()
                except Exception as exc:
                    log.warning("%s: stopping the preview failed: %s",
                                cam.info.serial, exc)
            self._started.clear()
            self.running = False

    def apply_live(self, cfg) -> None:
        """Exposure / gain / frame rate changes, without restarting the cameras."""
        self.cfg = cfg
        for cam in list(self._started):
            try:
                cam.apply_live(cfg)
            except Exception as exc:
                log.debug("%s: live setting change failed: %s", cam.info.serial, exc)

    # ---- inspection --------------------------------------------------------
    def preview(self, serial: str) -> Optional[np.ndarray]:
        with self._data_lock:
            return self._images.get(serial)

    def stats(self, serial: str) -> Dict[str, object]:
        with self._data_lock:
            return dict(self._stats.get(serial, {}))

    # ---- internals ---------------------------------------------------------
    def _grab_loop(self, cam: CameraBackend) -> None:
        serial = cam.info.serial
        window_start, window_count = time.perf_counter(), 0
        last_thumb = 0.0
        timeouts = 0
        while not self._stop_evt.is_set():
            try:
                frame = cam.retrieve(500)
            except Exception as exc:
                with self._data_lock:
                    self._stats[serial]["error"] = str(exc)
                log.error("%s: live preview stopped: %s", serial, exc)
                return
            now = time.perf_counter()
            if frame is None:
                timeouts += 1
                if timeouts == 10:
                    log.warning("%s: no image for 5 s in the live preview", serial)
                if now - window_start > 2.0:
                    with self._data_lock:
                        self._stats[serial]["fps"] = 0.0
                continue
            timeouts = 0
            window_count += 1
            with self._data_lock:
                st = self._stats[serial]
                st["grabbed"] = int(st["grabbed"]) + 1
                st["error"] = ""
                if now - window_start >= 1.0:
                    st["fps"] = window_count / (now - window_start)
                    window_start, window_count = now, 0
            if now - last_thumb >= 1.0 / PREVIEW_RATE_HZ:
                last_thumb = now
                thumb = make_thumbnail(frame.array)
                with self._data_lock:
                    self._images[serial] = thumb
