"""Camera discovery across backends."""
from __future__ import annotations

import logging
from typing import List, Tuple

from ..camera import CameraBackend

log = logging.getLogger(__name__)


def discover(cfg) -> Tuple[List[CameraBackend], str]:
    """Return (cameras, backend_name) honouring cfg.acquisition.backend.

    'auto' prefers real (or pylon-emulated) Basler devices and only falls back to the
    built-in dummies when none are present, so plugging the cameras in is all it takes
    to switch from bench testing to a real recording.
    """
    requested = cfg.acquisition.backend
    if requested in ("auto", "pylon"):
        try:
            from .pylon_backend import PYLON_AVAILABLE, discover_pylon
            if PYLON_AVAILABLE:
                cams = discover_pylon(cfg)
                if cams:
                    return _filter_enabled(cams, cfg), "pylon"
                if requested == "pylon":
                    raise RuntimeError(
                        "no Basler cameras found. Connect them, or set PYLON_CAMEMU=8 "
                        "to use the pylon emulator, or switch the backend to 'dummy'.")
                log.info("no Basler devices found, falling back to dummy cameras")
            elif requested == "pylon":
                raise RuntimeError("pypylon is not installed (pip install pypylon)")
            else:
                log.info("pypylon unavailable, falling back to dummy cameras")
        except ImportError as exc:
            if requested == "pylon":
                raise
            log.info("pylon backend unavailable (%s), using dummy cameras", exc)

    from .dummy_backend import discover_dummy
    return _filter_enabled(discover_dummy(cfg), cfg), "dummy"


def _filter_enabled(cams: List[CameraBackend], cfg) -> List[CameraBackend]:
    keep = [c for c in cams if cfg.override_for(c.info.serial).enabled]
    for c in cams:
        if c not in keep:
            log.info("skipping %s (disabled in config)", c.info.serial)
    return keep
