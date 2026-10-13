"""Basler cameras via pypylon.

Works against real GigE/USB3 cameras and against Basler's own camera emulator
(set PYLON_CAMEMU=8 before starting the app to get 8 emulated devices).

Basler exposes the same feature under different names depending on the camera
generation -- ace classic GigE uses the old GigE Vision names (`ExposureTimeAbs`,
`GevIEEE1588`), while ace 2 / dart / USB3 use SFNC names (`ExposureTime`,
`PtpEnable`). Every access here goes through `_first_node`, which picks whichever
name the connected camera actually has, so one code path covers both.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from ..camera import (Capability, CameraBackend, CameraError, CameraInfo, Frame,
                      INVALID_BLOCK_ID, now_ns)
from ..writer import genicam_pix_fmt

log = logging.getLogger(__name__)

try:
    from pypylon import genicam, pylon
    PYLON_AVAILABLE = True
except Exception as _exc:                                  # pragma: no cover
    genicam = pylon = None
    PYLON_AVAILABLE = False
    PYLON_IMPORT_ERROR = _exc

# Feature-name aliases, most modern name first.
NODES = {
    "exposure":        ("ExposureTime", "ExposureTimeAbs"),
    # In dB. GainRaw (ace classic) is in camera-specific units, not dB, so it is
    # handled separately in configure() rather than listed here.
    "gain":            ("Gain", "GainAbs"),
    "fps":             ("AcquisitionFrameRate", "AcquisitionFrameRateAbs"),
    "fps_enable":      ("AcquisitionFrameRateEnable", "AcquisitionFrameRateEnableAbs"),
    "ptp_enable":      ("PtpEnable", "GevIEEE1588"),
    "ptp_status":      ("PtpStatus", "GevIEEE1588Status", "GevIEEE1588StatusLatched"),
    "ptp_servo":       ("PtpServoStatus",),
    "ptp_latch":       ("PtpDataSetLatch", "GevIEEE1588DataSetLatch"),
    "ptp_offset":      ("PtpOffsetFromMaster", "GevIEEE1588OffsetFromMaster"),
    "ptp_clock_acc":   ("PtpClockAccuracy",),
    "ts_latch":        ("TimestampLatch", "GevTimestampControlLatch"),
    "ts_value":        ("TimestampLatchValue", "GevTimestampValue"),
    "ts_freq":         ("GevTimestampTickFrequency",),
    "packet_size":     ("GevSCPSPacketSize",),
    "packet_delay":    ("GevSCPD",),
    "resulting_fps":   ("ResultingFrameRate", "ResultingFrameRateAbs"),
    "payload_size":    ("PayloadSize",),
}


class PylonCamera(CameraBackend):
    def __init__(self, device_info):
        self._di = device_info
        self.info = CameraInfo(
            serial=device_info.GetSerialNumber(),
            model=device_info.GetModelName(),
            device_class=device_info.GetDeviceClass(),
            friendly_name=device_info.GetFriendlyName(),
            ip_address=(device_info.GetIpAddress()
                        if device_info.IsIpAddressAvailable() else ""),
            user_id=(device_info.GetUserDefinedName()
                     if device_info.IsUserDefinedNameAvailable() else ""),
        )
        self._cam: Optional["pylon.InstantCamera"] = None
        self._converter: Optional["pylon.ImageFormatConverter"] = None
        self._caps: Optional[set] = None
        self._index_counter = 0
        self._last_block = -1
        self._skipped = 0
        self._needs_conversion = False
        self._chunks_enabled = False

    # ---- node helpers ------------------------------------------------------
    def _nodemap(self):
        if self._cam is None:
            raise CameraError(f"{self.info.serial}: camera is not open")
        return self._cam.GetNodeMap()

    def _first_node(self, key_or_names, require_writable: bool = False):
        """Return the first available GenICam node from an alias group, else None."""
        names = NODES.get(key_or_names, None)
        if names is None:
            names = (key_or_names,) if isinstance(key_or_names, str) else tuple(key_or_names)
        nm = self._nodemap()
        for name in names:
            try:
                node = nm.GetNode(name)
            except Exception:
                node = None
            if node is None:
                continue
            try:
                if not genicam.IsAvailable(node):
                    continue
                if require_writable and not genicam.IsWritable(node):
                    continue
            except Exception:
                continue
            return node
        return None

    def _has(self, key) -> bool:
        return self._first_node(key) is not None

    def _read(self, key, default=None):
        node = self._first_node(key)
        if node is None:
            return default
        try:
            return node.GetValue()
        except Exception:
            try:
                return node.ToString()
            except Exception:
                return default

    def _write(self, key, value, clip: bool = True) -> bool:
        node = self._first_node(key, require_writable=True)
        if node is None:
            return False
        try:
            if clip and isinstance(value, (int, float)) and not isinstance(value, bool):
                value = self._clip(node, value)
            node.SetValue(value)
            return True
        except Exception as exc:
            log.debug("%s: could not set %s=%r (%s)", self.info.serial, key, value, exc)
            return False

    @staticmethod
    def _clip(node, value):
        """Clamp to the node's range and snap to its increment, where it has one."""
        try:
            lo, hi = node.GetMin(), node.GetMax()
        except Exception:
            return value
        value = min(max(value, lo), hi)
        try:
            inc = node.GetInc()
        except Exception:
            inc = None
        if inc:
            if isinstance(lo, int) and isinstance(inc, int):
                value = lo + ((int(value) - lo) // inc) * inc
            else:
                # Float node with a step (e.g. 0.1 us): snap without integer maths,
                # which would turn a sub-1 step into a division by zero.
                steps = round((float(value) - float(lo)) / float(inc))
                value = min(float(hi), float(lo) + steps * float(inc))
        return value

    def _execute(self, key) -> bool:
        node = self._first_node(key)
        if node is None:
            return False
        try:
            node.Execute()
            return True
        except Exception:
            return False

    def _enum_entries(self, name: str) -> List[str]:
        node = self._first_node(name)
        if node is None:
            return []
        try:
            return [e.GetSymbolic() for e in node.GetEntries() if genicam.IsAvailable(e)]
        except Exception:
            return []

    # ---- lifecycle ---------------------------------------------------------
    def open(self) -> None:
        if self._cam is not None and self._cam.IsOpen():
            return
        tlf = pylon.TlFactory.GetInstance()
        self._cam = pylon.InstantCamera(tlf.CreateDevice(self._di))
        self._cam.Open()
        self._caps = None

    def close(self) -> None:
        if self._cam is None:
            return
        try:
            if self._cam.IsGrabbing():
                self._cam.StopGrabbing()
            if self._cam.IsOpen():
                self._cam.Close()
        finally:
            self._cam = None

    def is_open(self) -> bool:
        return self._cam is not None and self._cam.IsOpen()

    # ---- capabilities ------------------------------------------------------
    def capabilities(self) -> set:
        if self._caps is not None:
            return self._caps
        caps = set()
        if not self.is_open():
            return caps
        sources = self._enum_entries("TriggerSource")
        if any(s == "Software" for s in sources):
            caps.add(Capability.SOFTWARE_TRIGGER)
        if any(s.startswith("Line") for s in sources):
            caps.add(Capability.HARDWARE_TRIGGER)
        # Action commands only actually work over GigE, whatever the node map says.
        if self.info.is_gige and any(s.startswith("Action") for s in sources):
            caps.add(Capability.ACTION_COMMAND)
        if self.info.is_gige and self._has("ptp_enable"):
            caps.add(Capability.PTP)
        if self._has("ChunkModeActive"):
            caps.add(Capability.CHUNK_TIMESTAMP)
        if self._enum_entries("LineSelector"):
            caps.add(Capability.LINE_OUTPUT)
        self._caps = caps
        return caps

    # ---- configuration -----------------------------------------------------
    def configure(self, cfg, serial_overrides=True) -> None:
        if not self.is_open():
            raise CameraError(f"{self.info.serial}: configure() before open()")
        acq = cfg.acquisition
        serial = self.info.serial
        roi = cfg.roi_for(serial) if serial_overrides else acq.roi

        self._write("AcquisitionMode", "Continuous")
        # Binning first: it changes the image's maximum size, which the ROI is cut from.
        self._apply_binning(int(getattr(acq, "binning", 1) or 1))
        # Offsets must go to zero before widening the ROI, or the writes are clipped.
        self._write("OffsetX", 0)
        self._write("OffsetY", 0)
        self._write("Width", int(roi.width))
        self._write("Height", int(roi.height))
        self._write("OffsetX", int(roi.offset_x))
        self._write("OffsetY", int(roi.offset_y))

        w, h = int(self._read("Width", 0) or 0), int(self._read("Height", 0) or 0)
        if (w, h) != (int(roi.width), int(roi.height)):
            log.info("%s: image size %dx%d (requested %dx%d; limited by the sensor%s)",
                     serial, w, h, roi.width, roi.height,
                     " after binning" if getattr(acq, "binning", 1) > 1 else "")
        if acq.pixel_format and acq.pixel_format in self._enum_entries("PixelFormat"):
            self._write("PixelFormat", acq.pixel_format)
        elif acq.pixel_format:
            log.warning("%s: pixel format %s unsupported, keeping %s",
                        serial, acq.pixel_format, self._read("PixelFormat"))

        self._write("ExposureAuto", "Off")
        self._write("GainAuto", "Off")
        self._apply_exposure_gain(cfg, serial_overrides)

        if self.info.is_gige:
            if acq.packet_size:
                self._write("packet_size", int(acq.packet_size))
            if acq.inter_packet_delay_us is not None:
                self._packet_delay_ns = float(acq.inter_packet_delay_us) * 1000.0
            else:
                self._packet_delay_ns = int(acq.inter_packet_delay) * 8.0
            self._apply_packet_delay()

        # Cap the free-run rate. set_trigger("on") switches this limiter off again.
        self._target_fps = float(acq.fps)
        if self._write("fps_enable", True):
            self._write("fps", float(acq.fps))

        # Done in the camera with ReverseX/ReverseY rather than in numpy: it is free,
        # and it keeps the recorded frames identical to what the sensor would produce
        # if the camera were mounted the right way up.
        if cfg.override_for(serial).rotate_180:
            if not (self._write("ReverseX", True) and self._write("ReverseY", True)):
                raise CameraError(
                    f"{serial}: rotate_180 is set but the camera has no "
                    f"ReverseX/ReverseY. Mount it upright or rotate downstream.")
        else:
            self._write("ReverseX", False)
            self._write("ReverseY", False)

        self._enable_chunk_timestamp()
        self._setup_converter()
        self._index_counter = 0
        self._last_block = -1
        self._skipped = 0

    def apply_live(self, cfg) -> None:
        if not self.is_open():
            return
        self._apply_exposure_gain(cfg, True)
        # Only while free-running: with triggers the limiter must stay off.
        if bool(self._read("fps_enable", False)):
            self._target_fps = float(cfg.acquisition.fps)
            self._write("fps", float(cfg.acquisition.fps))

    def _apply_exposure_gain(self, cfg, serial_overrides: bool) -> None:
        acq = cfg.acquisition
        serial = self.info.serial
        exposure = cfg.exposure_for(serial) if serial_overrides else acq.exposure_us
        gain = cfg.gain_for(serial) if serial_overrides else acq.gain_db
        if not self._write("exposure", float(exposure)):
            log.warning("%s: could not set exposure to %.0f us", serial, exposure)
        if self._has("gain"):
            self._write("gain", float(gain))
        elif self._has("GainRaw"):
            # Raw units differ per sensor; only "no gain" maps safely (the minimum).
            node = self._first_node("GainRaw", require_writable=True)
            if node is not None and float(gain) == 0.0:
                node.SetValue(node.GetMin())
            elif float(gain) != 0.0:
                log.warning("%s: this camera sets gain in raw units, not dB; the "
                            "configured %.1f dB is not applied (gain left at %s)",
                            serial, gain, self._read("GainRaw"))

    def binning_support(self) -> Dict[str, int]:
        """Largest binning / decimation factor the camera offers (1 = none)."""
        out = {}
        for key, names in (("binning", ("BinningHorizontal", "BinningVertical")),
                           ("decimation", ("DecimationHorizontal", "DecimationVertical"))):
            maxima = []
            for n in names:
                node = self._first_node(n)
                try:
                    maxima.append(int(node.GetMax()) if node is not None else 1)
                except Exception:
                    maxima.append(1)
            out[key] = min(maxima)
        return out

    def _apply_binning(self, factor: int) -> None:
        """Bin (or else decimate) by `factor` in both directions; 1 switches it off."""
        support = self.binning_support()
        if factor <= 1:
            for n in ("BinningHorizontal", "BinningVertical",
                      "DecimationHorizontal", "DecimationVertical"):
                if self._has(n):
                    self._write(n, 1)
            return
        if support["binning"] >= factor:
            # Average keeps the brightness of the unbinned image; Sum would raise it.
            for names in (("BinningHorizontalMode", "BinningModeHorizontal"),
                          ("BinningVerticalMode", "BinningModeVertical")):
                for n in names:
                    entries = self._enum_entries(n)
                    mode = next((e for e in entries if e.startswith("Averag")), None)
                    if mode:
                        self._write(n, mode)
                        break
            ok = self._write("BinningHorizontal", factor) and \
                self._write("BinningVertical", factor)
            if ok:
                log.info("%s: binning %dx%d (full field of view, 1/%d of the pixels)",
                         self.info.serial, factor, factor, factor * factor)
                return
        if support["decimation"] >= factor:
            if self._write("DecimationHorizontal", factor) and \
                    self._write("DecimationVertical", factor):
                log.info("%s: no binning on this camera; decimation %dx%d instead "
                         "(full field of view, every %d. pixel)",
                         self.info.serial, factor, factor, factor)
                return
        raise CameraError(
            f"{self.info.serial}: binning {factor} requested, but this camera supports "
            f"binning up to {support['binning']} and decimation up to "
            f"{support['decimation']} (colour models usually support neither). Set "
            f"acquisition.binning to 1; for the full field of view at full resolution "
            f"the network needs more bandwidth (see 'python -m syncrec check').")

    def _enable_chunk_timestamp(self) -> None:
        self._chunks_enabled = False
        if not self._has("ChunkModeActive"):
            return
        try:
            if not self._write("ChunkModeActive", True):
                return
            if "Timestamp" in self._enum_entries("ChunkSelector"):
                self._write("ChunkSelector", "Timestamp")
                self._write("ChunkEnable", True)
            self._chunks_enabled = True
        except Exception as exc:
            log.debug("%s: chunk timestamp unavailable (%s)", self.info.serial, exc)

    def _setup_converter(self) -> None:
        fmt = str(self._read("PixelFormat", "Mono8"))
        self._needs_conversion = fmt.startswith("Bayer") or fmt.endswith(("10", "12", "16"))
        if not self._needs_conversion:
            self._converter = None
            return
        self._converter = pylon.ImageFormatConverter()
        self._converter.OutputPixelFormat = (pylon.PixelType_BGR8packed
                                             if fmt.startswith("Bayer")
                                             else pylon.PixelType_Mono8)
        self._converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    def output_geometry(self):
        """Frame size and ffmpeg pixel format, after any debayering we apply."""
        width = int(self._read("Width", 0))
        height = int(self._read("Height", 0))
        if not width or not height:
            raise CameraError(f"{self.info.serial}: camera reports no image size")
        fmt = str(self._read("PixelFormat", "Mono8"))
        if self._needs_conversion and self._converter is not None:
            fmt = "BGR8Packed" if fmt.startswith("Bayer") else "Mono8"
        return width, height, genicam_pix_fmt(fmt)

    def get_param(self, name: str, default: Any = None) -> Any:
        return self._read(name, default)

    def set_param(self, name: str, value: Any) -> bool:
        return self._write(name, value)

    # ---- acquisition -------------------------------------------------------
    def start(self) -> None:
        if not self.is_open():
            raise CameraError(f"{self.info.serial}: start() before open()")
        self._cam.MaxNumBuffer = 64
        # Image timestamps come in device ticks; remember how to turn them into ns
        # (the tick length depends on the PTP state, which is final by now).
        freq = self._read("ts_freq", None)
        try:
            self._tick_ns = 1e9 / float(freq) if freq else 1.0
        except (TypeError, ValueError, ZeroDivisionError):
            self._tick_ns = 1.0
        if self.info.is_gige:
            self._apply_packet_delay()
            # pylon gives up on a frame FrameRetention ms after its first packet
            # (default 200). With an inter-packet delay, so several cameras can share
            # one link, a full frame can take about that long to arrive, and resent
            # packets come later still -- the frame is then dropped as incomplete
            # although every packet would have arrived. Allow far more time.
            self._set_stream_param("FrameRetention", 1000)
        self._check_frame_rate()
        try:
            self._cam.StartGrabbing(pylon.GrabStrategy_OneByOne)
        except Exception as exc:
            # Some devices advertise ChunkModeActive but ship no chunk node map
            # (the pylon emulator does this). Drop chunks and grab anyway.
            if not self._chunks_enabled:
                raise CameraError(f"{self.info.serial}: StartGrabbing failed: {exc}") from exc
            log.warning("%s: disabling chunk data after grab start failed (%s)",
                        self.info.serial, exc)
            self._write("ChunkModeActive", False)
            self._chunks_enabled = False
            self._cam.StartGrabbing(pylon.GrabStrategy_OneByOne)

    def _apply_packet_delay(self) -> None:
        """Write the inter-packet delay in the camera's current tick length.

        GevSCPD counts timestamp ticks, and the tick length depends on the PTP state
        (ace classic: 8 ns without PTP, 1 ns with it). So this is re-applied right
        before grabbing, once PTP is in its final state.
        """
        delay_ns = getattr(self, "_packet_delay_ns", None)
        if delay_ns is None or not self.info.is_gige:
            return
        freq = self._read("ts_freq", None)
        try:
            freq = float(freq) if freq else 125e6
        except (TypeError, ValueError):
            freq = 125e6
        ticks = int(round(delay_ns * freq / 1e9))
        self._write("packet_delay", ticks)
        log.info("%s: inter-packet delay %.0f us = %d ticks at %.0f MHz (camera has %s)",
                 self.info.serial, delay_ns / 1000, ticks, freq / 1e6,
                 self._read("packet_delay"))

    def _check_frame_rate(self) -> None:
        """Warn when the camera cannot produce the requested rate with these settings.

        A camera that is not ready ignores a trigger, so a limit below the trigger
        rate does not show up as an error: the camera just takes every 2nd or 3rd
        trigger, and the recording fills the gaps with placeholders.
        """
        fps = getattr(self, "_target_fps", 0.0)
        if not fps:
            return
        period_ms = 1000.0 / fps
        problems = []
        exposure_us = self._read("exposure", None)
        if exposure_us and float(exposure_us) / 1000.0 >= period_ms:
            problems.append(f"exposure {float(exposure_us) / 1000:.1f} ms is not shorter "
                            f"than the {period_ms:.1f} ms frame period")
        resulting = self._read("resulting_fps", None)
        if resulting:
            log.info("%s: camera reports a maximum of %.1f fps with these settings",
                     self.info.serial, float(resulting))
            if float(resulting) < fps * 0.999:
                problems.append(f"camera reports at most {float(resulting):.1f} fps "
                                f"with the current exposure / ROI / bandwidth settings")
        delay_ns = getattr(self, "_packet_delay_ns", None)
        payload = self._read("payload_size", None)
        packet = self._read("packet_size", None)
        if self.info.is_gige and delay_ns is not None and payload and packet:
            data_per_packet = max(1, int(packet) - 36)            # minus GVSP headers
            n_packets = -(-int(payload) // data_per_packet)
            wire_ns = (int(packet) + 38) * 8.0                   # 1 Gb/s camera link
            transfer_ms = n_packets * (wire_ns + delay_ns) / 1e6
            log.info("%s: sending one frame takes about %.1f ms (%d packets)",
                     self.info.serial, transfer_ms, n_packets)
            if transfer_ms > period_ms:
                problems.append(f"sending one frame takes about {transfer_ms:.0f} ms "
                                f"with the inter-packet delay, longer than the "
                                f"{period_ms:.1f} ms frame period")
        for p in problems:
            log.warning("%s: cannot keep up with %.1f fps: %s. The camera will ignore "
                        "triggers it is not ready for.", self.info.serial, fps, p)

    def _set_stream_param(self, name: str, value) -> bool:
        try:
            node = self._cam.GetStreamGrabberNodeMap().GetNode(name)
            if node is None or not genicam.IsWritable(node):
                return False
            node.SetValue(value)
            return True
        except Exception as exc:
            log.debug("%s: could not set stream %s=%r (%s)", self.info.serial,
                      name, value, exc)
            return False

    def stop(self) -> None:
        if self._cam is not None and self._cam.IsGrabbing():
            self._cam.StopGrabbing()

    def is_grabbing(self) -> bool:
        return self._cam is not None and self._cam.IsGrabbing()

    def retrieve(self, timeout_ms: int) -> Optional[Frame]:
        if self._cam is None:
            return None
        try:
            res = self._cam.RetrieveResult(int(timeout_ms), pylon.TimeoutHandling_Return)
        except Exception as exc:
            raise CameraError(f"{self.info.serial}: RetrieveResult failed: {exc}") from exc
        if res is None or not res.IsValid():
            return None
        try:
            if not res.GrabSucceeded():
                self._skipped += 1
                # Count this frame once: remember its block ID so the gap check on
                # the next good frame does not count it a second time.
                try:
                    failed_block = int(res.BlockID)
                    if failed_block != INVALID_BLOCK_ID:
                        self._last_block = failed_block
                except Exception:
                    pass
                log.warning("%s: grab failed (%s)", self.info.serial,
                            res.GetErrorDescription())
                return None
            if self._needs_conversion and self._converter is not None:
                arr = self._converter.Convert(res).GetArray()
            else:
                arr = res.GetArray()
            arr = np.ascontiguousarray(arr)

            block = int(res.BlockID)
            skipped = self._skipped
            if block != INVALID_BLOCK_ID and self._last_block >= 0:
                gap = block - self._last_block - 1
                if gap > 0:
                    skipped += gap
            if block != INVALID_BLOCK_ID:
                self._last_block = block

            cam_ts = self._result_timestamp(res)
            frame = Frame(serial=self.info.serial, index=self._index_counter, array=arr,
                          host_timestamp_ns=now_ns(), cam_timestamp_ns=cam_ts,
                          block_id=block, skipped_before=skipped,
                          pixel_format=str(self._read("PixelFormat", "Mono8")))
            self._index_counter += 1
            self._skipped = 0
            return frame
        finally:
            res.Release()

    def _result_timestamp(self, res) -> int:
        """Device timestamp in nanoseconds, 0 if the camera does not provide one."""
        for attr in ("ChunkTimestamp", "TimeStamp"):
            try:
                val = getattr(res, attr)
                val = int(val.GetValue() if hasattr(val, "GetValue") else val)
            except Exception:
                continue
            if val and val != INVALID_BLOCK_ID:
                scale = getattr(self, "_tick_ns", 1.0)
                return int(val * scale) if scale != 1.0 else val
        return 0

    def execute_software_trigger(self) -> None:
        if self._cam is None:
            raise CameraError(f"{self.info.serial}: not open")
        self._cam.ExecuteSoftwareTrigger()

    def wait_for_trigger_ready(self, timeout_ms: int = 1000) -> bool:
        try:
            return bool(self._cam.WaitForFrameTriggerReady(
                int(timeout_ms), pylon.TimeoutHandling_Return))
        except Exception:
            return True

    # ---- sync --------------------------------------------------------------
    def set_trigger(self, mode: str, source: Optional[str] = None,
                    activation: Optional[str] = None) -> None:
        """Turn every trigger off, then arm FrameStart only. Avoids stale trigger state."""
        selectors = self._enum_entries("TriggerSelector") or ["FrameStart"]
        for sel in selectors:
            if self._write("TriggerSelector", sel):
                self._write("TriggerMode", "Off")

        if mode != "on":
            self._write("fps_enable", True)
            return

        target = "FrameStart" if "FrameStart" in selectors else selectors[0]
        if not self._write("TriggerSelector", target):
            raise CameraError(f"{self.info.serial}: cannot select trigger {target}")
        if source:
            available = self._enum_entries("TriggerSource")
            if source not in available:
                raise CameraError(
                    f"{self.info.serial}: trigger source {source!r} not available "
                    f"(has: {', '.join(available) or 'none'})")
            self._write("TriggerSource", source)
        if activation and activation in self._enum_entries("TriggerActivation"):
            self._write("TriggerActivation", activation)
        elif activation:
            log.debug("%s: trigger activation %r not available", self.info.serial,
                      activation)
        self._write("TriggerMode", "On")
        # With triggers the trigger sets the rate. Leaving the frame-rate limiter on
        # at the same rate makes the camera ignore any trigger that arrives a hair
        # early by its own oscillator -- i.e. silently drop frames.
        self._write("fps_enable", False)

    def configure_line_output(self, line: str, source: str) -> bool:
        """Make this camera drive `line` so it can trigger the others."""
        if not self._write("LineSelector", line):
            return False
        self._write("LineMode", "Output")
        ok = self._write("LineSource", source)
        return bool(ok)

    def configure_ptp(self, enable: bool) -> None:
        if not self._has("ptp_enable"):
            if enable:
                raise CameraError(f"{self.info.serial}: camera has no PTP support")
            return
        # Rewriting the enable bit can restart the camera's PTP state machine and
        # throw away a lock it already has, so only write on an actual change.
        if bool(self._read("ptp_enable", False)) == bool(enable):
            return
        self._write("ptp_enable", bool(enable))

    def ptp_status(self) -> Dict[str, Any]:
        if not self._has("ptp_enable"):
            return {"enabled": False, "state": "unsupported", "offset_ns": None}
        enabled = bool(self._read("ptp_enable", False))
        self._execute("ptp_latch")
        state = self._read("ptp_status", None) or self._read("ptp_servo", None) or "Unknown"
        offset = self._read("ptp_offset", None)
        return {"enabled": enabled, "state": str(state),
                "offset_ns": int(offset) if offset is not None else None}

    def device_timestamp_ns(self) -> Optional[int]:
        """Current device time in ns -- the unit scheduled action commands use.

        GigE cameras report timestamps in ticks. ace classic runs at 125 MHz without
        PTP and 1 GHz with it; SFNC cameras (ace 2) have no tick node and are already
        in ns. Convert rather than assume.
        """
        if not self._execute("ts_latch"):
            return None
        val = self._read("ts_value", None)
        if val is None:
            return None
        freq = self._read("ts_freq", None)
        try:
            freq = int(freq) if freq is not None else 0
        except (TypeError, ValueError):
            freq = 0
        if freq and freq != 1_000_000_000:
            return int(int(val) * (1_000_000_000 / freq))
        return int(val)

    def action_queue_size(self) -> Optional[int]:
        """How many scheduled action commands the camera can hold at once."""
        val = self._read(("ActionQueueSize", "GevActionQueueSize"), None)
        try:
            return int(val) if val is not None else None
        except (TypeError, ValueError):
            return None

    def subnet_broadcast(self) -> Optional[str]:
        """Directed broadcast address of this camera's subnet, e.g. 169.254.255.255."""
        di = self._di
        try:
            if not (di.IsIpAddressAvailable() and di.IsSubnetMaskAvailable()):
                return None
            ip = [int(x) for x in di.GetIpAddress().split(".")]
            mask = [int(x) for x in di.GetSubnetMask().split(".")]
        except Exception:
            return None
        if len(ip) != 4 or len(mask) != 4:
            return None
        return ".".join(str(i | (~m & 0xFF)) for i, m in zip(ip, mask))

    def configure_action_keys(self, device_key: int, group_key: int,
                              group_mask: int) -> None:
        # ActionSelector is an integer on GigE cameras, not an enumeration. The keys
        # below are stored per selector, and TriggerSource "Action1" only listens to
        # selector 1 -- keys written to another index are acknowledged but never fire.
        if self._has("ActionSelector") and not self._write("ActionSelector", 1):
            raise CameraError(f"{self.info.serial}: could not select action 1")
        ok = self._write("ActionDeviceKey", int(device_key))
        ok = self._write("ActionGroupKey", int(group_key)) and ok
        ok = self._write("ActionGroupMask", int(group_mask)) and ok
        if not ok:
            raise CameraError(f"{self.info.serial}: could not write action command keys")

    def statistics(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {}
        if self._cam is None:
            return stats
        for name, key in (("failed_buffers", "Statistic_Failed_Buffer_Count"),
                          ("buffer_underruns", "Statistic_Buffer_Underrun_Count"),
                          ("resend_requests", "Statistic_Resend_Request_Count"),
                          ("missed_frames", "Statistic_Missed_Frame_Count")):
            try:
                node = self._cam.GetStreamGrabberNodeMap().GetNode(key)
                if node is not None and genicam.IsReadable(node):
                    stats[name] = int(node.GetValue())
            except Exception:
                pass
        return stats


def gige_transport_layer():
    """The GigE transport layer, needed to broadcast action commands. None if absent."""
    if not PYLON_AVAILABLE:
        return None
    try:
        return pylon.TlFactory.GetInstance().CreateTl(pylon.BaslerGigEDeviceClass)
    except Exception as exc:
        log.debug("no GigE transport layer: %s", exc)
        return None


def discover_pylon(cfg=None) -> List[PylonCamera]:
    """Enumerate every attached Basler device (including emulated ones)."""
    if not PYLON_AVAILABLE:
        raise CameraError(f"pypylon is not installed: {PYLON_IMPORT_ERROR}")
    devices = pylon.TlFactory.GetInstance().EnumerateDevices()
    cams = [PylonCamera(d) for d in devices]
    cams.sort(key=lambda c: c.info.serial)
    return cams
