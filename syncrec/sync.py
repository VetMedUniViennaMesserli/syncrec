"""Frame-synchronisation strategies.

Four ways to make N cameras expose at the same instant, best first:

  ptp_action       GigE only. The cameras discipline their clocks to each other with
                   IEEE 1588 (PTP), then the host broadcasts *scheduled* action
                   commands: "everyone expose at PTP time T". Accuracy is the PTP
                   residual, typically well under 100 us, and the host's own timing
                   jitter does not matter because T is decided in advance.
  hardware_trigger One electrical signal into every camera's trigger input, either
                   from an external generator or from one camera acting as master.
                   The most accurate option and the only one that works on USB3.
  software_trigger The host tells each camera to expose, one call after another.
                   Accuracy is limited by per-camera command latency (sub-ms on GigE,
                   worse on USB3). Fine for testing, not for fast motion.
  freerun          No trigger at all: cameras run on their own oscillators and drift
                   apart by milliseconds within seconds. Recorded anyway, with
                   timestamps, so frames can be matched up afterwards -- but this is
                   a fallback, not a sync mode.

A strategy configures the cameras (`prepare`), and if it is host-paced the session
calls `emit_trigger` once per frame period.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .camera import Capability, CameraBackend

log = logging.getLogger(__name__)


class SyncError(RuntimeError):
    pass


@dataclass
class SyncStatus:
    mode: str
    ready: bool = False
    detail: str = ""
    per_camera: Dict[str, Any] = field(default_factory=dict)


class SyncStrategy:
    """Base class. Subclasses configure cameras and optionally pace the triggers."""

    name = "base"
    host_paced = False          # True -> session calls emit_trigger() every period
    # True when the sync mode can tell which trigger every frame answers (slot_for);
    # then a frame without a slot cannot be placed in the video and is left out.
    provides_slots = False

    def __init__(self, cfg):
        self.cfg = cfg
        self.cameras: List[CameraBackend] = []

    # ---- lifecycle ---------------------------------------------------------
    def prepare(self, cameras: Sequence[CameraBackend],
                progress: Optional[Callable[[str], None]] = None) -> None:
        """Configure trigger mode / PTP. May block while clocks converge."""
        self.cameras = list(cameras)

    def on_started(self) -> None:
        """Called once all cameras are grabbing."""

    def emit_trigger(self, trigger_id: int, due_ns: int) -> None:
        """Fire frame `trigger_id`. Only called when host_paced is True."""

    def slot_for(self, frame) -> Optional[int]:
        """Which trigger a frame answers, if this sync mode can tell. Else None."""
        return None

    def due_time(self, trigger_id: int) -> Optional[float]:
        """When (time.perf_counter seconds) to send trigger `trigger_id`.

        None: the session paces triggers on the host clock, one period apart.
        """
        return None

    @property
    def shared_device_clock(self) -> bool:
        """Do the cameras' timestamps run on one common clock?

        Only then can exposure timestamps be compared across cameras. Without PTP,
        every camera counts from its own power-up and the numbers are unrelated.
        Simulated cameras share the host clock.
        """
        return bool(self.cameras) and all(hasattr(c, "fire_external_trigger")
                                          for c in self.cameras)

    def teardown(self) -> None:
        for cam in self.cameras:
            try:
                cam.set_trigger("off")
            except Exception:
                pass

    def status(self) -> SyncStatus:
        return SyncStatus(mode=self.name, ready=True)

    # ---- helpers -----------------------------------------------------------
    @staticmethod
    def _simulate(cam: CameraBackend, trigger_id: int, ts_ns: int) -> bool:
        """Deliver a trigger to a dummy camera. Real cameras get it over the wire."""
        fire = getattr(cam, "fire_external_trigger", None)
        if fire is None:
            return False
        fire(trigger_id, ts_ns)
        return True


class FreeRunSync(SyncStrategy):
    name = "freerun"
    host_paced = False

    def prepare(self, cameras, progress=None) -> None:
        super().prepare(cameras, progress)
        for cam in self.cameras:
            cam.set_trigger("off")
        if progress:
            progress("Free run: cameras are NOT frame-synchronised, only timestamped.")

    def status(self) -> SyncStatus:
        return SyncStatus(mode=self.name, ready=True,
                          detail="no trigger - frames must be aligned afterwards")


class SoftwareTriggerSync(SyncStrategy):
    name = "software_trigger"
    host_paced = True

    def prepare(self, cameras, progress=None) -> None:
        super().prepare(cameras, progress)
        for cam in self.cameras:
            if not cam.has(Capability.SOFTWARE_TRIGGER):
                raise SyncError(f"{cam.info.serial} has no software trigger")
            cam.set_trigger("on", source="Software")
        if progress:
            progress(f"Software trigger armed on {len(self.cameras)} cameras "
                     f"(accuracy limited by host latency).")

    def emit_trigger(self, trigger_id: int, due_ns: int) -> None:
        # Two passes: get every camera ready first, then fire them back to back, so
        # the spread is only the trigger calls and not the readiness checks.
        for cam in self.cameras:
            cam.wait_for_trigger_ready(200)
        for cam in self.cameras:
            if self._simulate(cam, trigger_id, time.time_ns()):
                continue
            try:
                cam.execute_software_trigger()
            except Exception as exc:
                log.warning("%s: software trigger failed: %s", cam.info.serial, exc)

    def status(self) -> SyncStatus:
        return SyncStatus(mode=self.name, ready=True,
                          detail="host-paced software trigger (sub-ms to ms spread)")


class HardwareTriggerSync(SyncStrategy):
    """Every camera waits on an electrical trigger line.

    The signal comes either from external hardware (`master_serial` unset) or from one
    of the cameras, which free-runs at the configured frame rate and drives its output
    line for the others.
    """

    name = "hardware_trigger"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.master: Optional[CameraBackend] = None
        self._hw = cfg.sync.hardware

    @property
    def host_paced(self) -> bool:                       # type: ignore[override]
        # With a camera as master nothing needs pacing; with an external generator
        # nothing does either. Dummy cameras are the exception: they have no wires,
        # so the host has to stand in for the signal generator.
        return all(hasattr(c, "fire_external_trigger") for c in self.cameras) \
            if self.cameras else False

    def prepare(self, cameras, progress=None) -> None:
        super().prepare(cameras, progress)
        hw = self._hw
        if hw.master_serial:
            matches = [c for c in self.cameras if c.info.serial == hw.master_serial]
            if not matches:
                raise SyncError(f"master camera {hw.master_serial} is not connected")
            self.master = matches[0]

        for cam in self.cameras:
            if cam is self.master:
                continue
            if not cam.has(Capability.HARDWARE_TRIGGER):
                raise SyncError(f"{cam.info.serial} exposes no hardware trigger line")
            cam.set_trigger("on", source=hw.trigger_source,
                            activation=hw.trigger_activation)

        if self.master is not None:
            configure_line = getattr(self.master, "configure_line_output", None)
            if configure_line is None or not configure_line(hw.master_line,
                                                            hw.master_line_source):
                raise SyncError(
                    f"{self.master.info.serial}: cannot drive {hw.master_line} "
                    f"from {hw.master_line_source}")
            self.master.set_trigger("off")              # master free-runs at cfg fps
            if progress:
                progress(f"Master {self.master.info.serial} drives {hw.master_line} "
                         f"({hw.master_line_source}); {len(self.cameras) - 1} slaves on "
                         f"{hw.trigger_source}.")
        elif progress:
            progress(f"All {len(self.cameras)} cameras armed on {hw.trigger_source} "
                     f"({hw.trigger_activation}) - feed them your trigger signal.")

    def emit_trigger(self, trigger_id: int, due_ns: int) -> None:
        # Simulation path only: one shared timestamp for all cameras, which is what a
        # real trigger wire achieves.
        for cam in self.cameras:
            self._simulate(cam, trigger_id, due_ns)

    def status(self) -> SyncStatus:
        detail = (f"master {self.master.info.serial}" if self.master
                  else f"external signal on {self._hw.trigger_source}")
        return SyncStatus(mode=self.name, ready=True, detail=detail)


class PtpActionSync(SyncStrategy):
    """PTP-disciplined clocks + scheduled action commands (GigE only)."""

    name = "ptp_action"
    host_paced = True
    provides_slots = True
    shared_device_clock = True          # that is what PTP is for

    MONITOR_INTERVAL_S = 10.0

    def __init__(self, cfg):
        super().__init__(cfg)
        self.ptp = cfg.sync.ptp
        self._tl = None
        self._t_ref_ns = 0          # camera-clock time of trigger 0
        self._period_ns = int(round(1e9 / cfg.acquisition.fps))
        self._last_status: Dict[str, Any] = {}
        self._simulated = False
        self._late = 0
        self._sched_late = 0
        self._addresses: List[str] = []
        self._ack_problem = ""
        # Host <-> camera clock model: camera_ns = cam + (host_ns - host) * rate.
        # The trigger loop runs on the host clock, the action times on the PTP clock;
        # the two crystals drift apart (tens of ppm = seconds per day), so the model
        # is re-measured every MONITOR_INTERVAL_S to keep commands one lead time early.
        self._anchor = (0, 0)
        self._first = (0, 0)
        self._rate = 1.0
        self._issued = -1           # highest trigger id sent so far
        self._bad_slots = 0
        self._clock_steps = 0
        self._health = ""
        self._stop_monitor = threading.Event()
        self._lead_ns = int(self.ptp.action_lead_time_ms * 1e6)
        self._queue_size: Optional[int] = None
        self._rejected = 0
        self._ack_checks = 0

    def prepare(self, cameras, progress=None) -> None:
        super().prepare(cameras, progress)
        self._simulated = all(hasattr(c, "fire_external_trigger") for c in self.cameras)

        for cam in self.cameras:
            if not cam.has(Capability.PTP):
                raise SyncError(f"{cam.info.serial} does not support PTP "
                                f"(PTP sync needs GigE cameras)")
            if not cam.has(Capability.ACTION_COMMAND):
                raise SyncError(f"{cam.info.serial} does not support action commands")
            cam.configure_ptp(True)

        self._await_ptp_lock(progress)
        self._choose_lead_time(progress)

        for cam in self.cameras:
            cam.configure_action_keys(self.ptp.device_key, self.ptp.group_key,
                                      self.ptp.group_mask)
            cam.set_trigger("on", source="Action1")

        if not self._simulated:
            from .backends.pylon_backend import gige_transport_layer
            self._tl = gige_transport_layer()
            if self._tl is None:
                raise SyncError("no GigE transport layer available for action commands")
            self._addresses = self._resolve_broadcast()
        if progress:
            where = f" via {', '.join(self._addresses)}" if self._addresses else ""
            progress(f"PTP locked, action commands armed on {len(self.cameras)} "
                     f"cameras{where}.")

    def _choose_lead_time(self, progress=None) -> None:
        """Send each command early, but never more at once than the cameras can hold.

        A command for trigger n goes out `lead` before its exposure, so about
        lead / period commands are pending in the camera at any moment. A camera
        holds only a few scheduled actions (ace classic: one, if it does not say);
        commands beyond that are rejected, and the camera then takes only every
        2nd or 3rd trigger -- with 100 ms lead at 25 fps exactly every 3rd.
        """
        sizes = []
        for cam in self.cameras:
            getter = getattr(cam, "action_queue_size", None)
            size = getter() if getter else None
            if size:
                sizes.append(size)
        queue = min(sizes) if len(sizes) == len(self.cameras) else 1
        self._queue_size = queue
        # Leave a quarter period so the next command never overlaps the pending one.
        limit = int(queue * self._period_ns - self._period_ns / 4)
        configured = int(self.ptp.action_lead_time_ms * 1e6)
        self._lead_ns = max(1_000_000, min(configured, limit))
        msg = (f"action commands sent {self._lead_ns / 1e6:.0f} ms ahead (cameras queue "
               f"{queue} scheduled action{'s' if queue != 1 else ''}"
               f"{'' if len(sizes) == len(self.cameras) else ' - assumed'})")
        log.info("%s", msg)
        if progress:
            progress(msg)
        if self._lead_ns < 10_000_000:
            log.warning("only %.1f ms between sending a trigger and its exposure: a "
                        "busy PC may send some too late", self._lead_ns / 1e6)

    def _resolve_broadcast(self) -> List[str]:
        """Where to send action commands.

        The limited broadcast 255.255.255.255 is not guaranteed to leave through the
        adapter the cameras are on: on Windows it typically goes out of one interface
        only, chosen by route metric. With cameras on link-local 169.254.x.x addresses
        (which every unconfigured adapter on the PC also has) that is very often the
        wrong one, and the cameras simply never hear the trigger. Basler's own samples
        use the directed broadcast of the cameras' subnet instead, so do the same
        unless the config names a specific address.
        """
        configured = (self.ptp.broadcast_address or "").strip()
        if configured and configured not in ("auto", "255.255.255.255"):
            return [configured]
        found: List[str] = []
        for cam in self.cameras:
            getter = getattr(cam, "subnet_broadcast", None)
            addr = getter() if getter else None
            if addr and addr not in found:
                found.append(addr)
        if not found:
            log.warning("could not derive the cameras' subnet broadcast address; "
                        "falling back to 255.255.255.255")
            return ["255.255.255.255"]
        if len(found) > 1:
            log.warning("cameras are on %d different subnets (%s); PTP sync needs "
                        "them on one network", len(found), ", ".join(found))
        return found

    def _await_ptp_lock(self, progress=None) -> None:
        """Block until one camera is Master, the rest are Slaves, and offsets settle.

        Settled means every slave's offset to the master stays within max_offset_ns
        continuously for settle_s seconds (one good sample can be a fluke). The master's
        own offset register is ignored: it is meaningless, and can hold a stale value
        from before the camera won the master election. If the cameras are still
        locked from a previous recording this returns after settle_s.
        """
        settle_s = float(getattr(self.ptp, "settle_s", 5.0))
        start = time.time()
        deadline = start + self.ptp.sync_timeout_s
        last_report = 0.0
        good_since = None
        while True:
            states = {c.info.serial: c.ptp_status() for c in self.cameras}
            self._last_status = states
            roles = [str(s.get("state", "")) for s in states.values()]
            # Exact match: "PreMaster" is not a master yet.
            masters = sum(1 for r in roles if r == "Master")
            slaves = sum(1 for r in roles if r == "Slave")
            offsets = [abs(s["offset_ns"]) for s in states.values()
                       if str(s.get("state", "")) == "Slave"
                       and s.get("offset_ns") is not None]
            worst = max(offsets) if offsets else None
            good = (masters == 1 and slaves == len(self.cameras) - 1
                    and (worst is None or worst <= self.ptp.max_offset_ns))
            now = time.time()
            if not good:
                good_since = None
            elif good_since is None:
                good_since = now
            if good_since is not None and now - good_since >= settle_s:
                if progress:
                    progress(f"PTP converged after {time.time() - start:.0f}s: 1 master, "
                             f"{slaves} slaves, worst offset {worst} ns.")
                return
            if time.time() > deadline:
                raise SyncError(
                    f"PTP did not converge in {self.ptp.sync_timeout_s:.0f}s "
                    f"(masters={masters}, slaves={slaves}, worst offset={worst} ns, "
                    f"target {self.ptp.max_offset_ns} ns). If the offset was still "
                    f"falling, raise sync.ptp.sync_timeout_s or max_offset_ns; if "
                    f"roles never settled, check that all cameras are on one switch.")
            if progress and time.time() - last_report > 2.0:
                last_report = time.time()
                progress(f"waiting for PTP ({time.time() - start:.0f}s): {masters} "
                         f"master / {slaves} slave, worst offset {worst} ns "
                         f"(target {self.ptp.max_offset_ns} ns)")
            time.sleep(0.25)

    def on_started(self) -> None:
        """Pick the shared time base: schedule trigger 0 one lead time from now."""
        lead_ns = self._lead_ns
        ref = None
        for cam in self.cameras:
            getter = getattr(cam, "device_timestamp_ns", None)
            if getter is not None:
                ref = getter()
                if ref:
                    break
        if not ref:
            # PTP-locked cameras follow the grandmaster's clock; with no readable
            # device timestamp, host time is the best available stand-in.
            ref = time.time_ns()
        host = time.perf_counter_ns()
        self._t_ref_ns = ref + lead_ns
        self._anchor = self._first = (host, ref)
        self._rate = 1.0
        self._late = 0
        self._sched_late = 0
        self._issued = -1
        if not self._simulated:
            self._stop_monitor.clear()
            threading.Thread(target=self._monitor_loop, name="ptp-monitor",
                             daemon=True).start()

    # ---- clock model -------------------------------------------------------
    def _cam_now(self, host_ns: int) -> float:
        h, c = self._anchor
        return c + (host_ns - h) * self._rate

    def due_time(self, trigger_id: int) -> Optional[float]:
        if self._simulated or not self._t_ref_ns:
            return None
        # Host time at which the camera clock reads (action time - lead time).
        lead_ns = self._lead_ns
        h, c = self._anchor
        target_cam = self._t_ref_ns + trigger_id * self._period_ns - lead_ns
        return (h + (target_cam - c) / self._rate) / 1e9

    def _recalibrate(self) -> None:
        cam = self.cameras[0]
        getter = getattr(cam, "device_timestamp_ns", None)
        if getter is None:
            return
        h1 = time.perf_counter_ns()
        c = getter()
        h2 = time.perf_counter_ns()
        if not c or h2 - h1 > 5_000_000:            # slow read: too imprecise, skip
            return
        h = (h1 + h2) // 2
        err = c - self._cam_now(h)
        if abs(err) > 50e6:
            # The PTP time base itself jumped -- typically a camera rebooted and won
            # the master election with a fresh clock. Move the whole schedule with
            # it, so trigger numbers (and the video frame numbers) simply continue.
            self._t_ref_ns += int(err)
            self._anchor = self._first = (h, c)
            self._clock_steps += 1
            log.warning("PTP time jumped by %+.1f ms (a camera restarted or the PTP "
                        "master changed); the trigger schedule follows the new time "
                        "base", err / 1e6)
            return
        h0, c0 = self._first
        if h - h0 > 30e9:                            # long baseline -> stable rate
            self._rate = (c - c0) / (h - h0)
        self._anchor = (h, c)

    def _check_health(self) -> None:
        states = {}
        for cam in self.cameras:
            try:
                states[cam.info.serial] = cam.ptp_status()
            except Exception as exc:
                states[cam.info.serial] = {"state": f"error: {exc}", "offset_ns": None}
        self._last_status = states
        problems = []
        roles = [str(s.get("state", "")) for s in states.values()]
        if roles.count("Master") != 1 or roles.count("Slave") != len(roles) - 1:
            problems.append("roles " + ", ".join(f"{k}={v.get('state')}"
                                                 for k, v in states.items()))
        offsets = [abs(s["offset_ns"]) for s in states.values()
                   if s.get("state") == "Slave" and isinstance(s.get("offset_ns"), int)]
        limit = max(10 * self.ptp.max_offset_ns, 10_000)
        if offsets and max(offsets) > limit:
            problems.append(f"worst offset {max(offsets)} ns")
        health = "; ".join(problems)
        if health and health != self._health:
            log.warning("PTP sync degraded during recording: %s", health)
        elif self._health and not health:
            log.info("PTP sync healthy again")
        self._health = health

    def _monitor_loop(self) -> None:
        while not self._stop_monitor.wait(self.MONITOR_INTERVAL_S):
            try:
                self._recalibrate()
                self._check_health()
            except Exception as exc:
                log.debug("PTP monitor: %s", exc)

    def slot_for(self, frame) -> Optional[int]:
        # Every camera's clock follows the PTP master and trigger n is scheduled at
        # t_ref + n * period, so the exposure timestamp says which trigger it was --
        # even when frames before it were lost.
        ts = frame.cam_timestamp_ns
        if not ts or not self._t_ref_ns:
            return None
        slot = int(round((ts - self._t_ref_ns) / self._period_ns))
        # Plausibility: a frame cannot answer a trigger not yet sent, and it cannot be
        # more than a few seconds old. Anything else is a bad timestamp or a clock
        # jump; placing it would shift or wreck the video, so it is left out.
        if self._simulated or self._issued - int(10 * 1e9 / self._period_ns) <= slot \
                <= self._issued + 1:
            return slot
        self._bad_slots += 1
        if self._bad_slots in (1, 10, 100, 1000):
            log.warning("%s: frame timestamp does not match any recent trigger "
                        "(slot %d, last trigger %d); left out (%d so far)",
                        frame.serial, slot, self._issued, self._bad_slots)
        return None

    def emit_trigger(self, trigger_id: int, due_ns: int) -> None:
        action_time = self._t_ref_ns + trigger_id * self._period_ns
        self._issued = max(self._issued, trigger_id)
        if not self._simulated and self._anchor[0]:
            # Estimated camera time now. A command that arrives after its action time
            # no longer fires at the planned moment, so count those.
            cam_now = self._cam_now(time.perf_counter_ns())
            if action_time < cam_now:
                self._sched_late += 1
                if self._sched_late in (1, 10, 100, 1000):
                    log.warning("action command %d sent %.1f ms after its action time "
                                "(%d so far); the host is too slow to schedule ahead",
                                trigger_id, (cam_now - action_time) / 1e6,
                                self._sched_late)
        if self._simulated:
            for cam in self.cameras:
                self._simulate(cam, trigger_id, action_time)
            return
        if trigger_id == 0:
            # Send the first trigger with acknowledgements requested, so a trigger
            # that never reaches the cameras is reported instead of silently
            # producing a recording of nothing.
            self._issue_checked(int(action_time))
            return
        if trigger_id % 250 == 0:
            # Every 10 s (at 25 fps): ask for acknowledgements once, to catch
            # commands the cameras reject (e.g. their action queue is full).
            self._spot_check(trigger_id, int(action_time))
            return
        for addr in self._addresses:
            try:
                self._tl.IssueScheduledActionCommandNoWait(
                    self.ptp.device_key, self.ptp.group_key, self.ptp.group_mask,
                    int(action_time), addr)
            except Exception as exc:
                self._late += 1
                if self._late <= 5:
                    log.error("scheduled action command %d failed: %s", trigger_id, exc)

    def _spot_check(self, trigger_id: int, action_time: int) -> None:
        n = len(self.cameras)
        bad = {}
        for addr in self._addresses:
            try:
                _ok, results = self._tl.IssueScheduledActionCommandWait(
                    self.ptp.device_key, self.ptp.group_key, self.ptp.group_mask,
                    action_time, addr, 30, n)
            except Exception as exc:
                log.debug("acknowledged action command %d failed: %s", trigger_id, exc)
                return
            for ip, status in results or ():
                if ip and ip != "0.0.0.0" and int(status) != 0:
                    bad[ip] = int(status)
        self._ack_checks += 1
        if bad:
            self._rejected += 1
            if self._rejected in (1, 10, 100):
                log.warning("cameras rejected action command %d: %s. If this repeats, "
                            "lower sync.ptp.action_lead_time_ms", trigger_id,
                            ", ".join(f"{ip} status 0x{st:04X}" for ip, st in bad.items()))

    def _issue_checked(self, action_time: int) -> None:
        n = len(self.cameras)
        acked: Dict[str, int] = {}
        for addr in self._addresses:
            try:
                _ok, results = self._tl.IssueScheduledActionCommandWait(
                    self.ptp.device_key, self.ptp.group_key, self.ptp.group_mask,
                    action_time, addr, 250, n)
            except Exception as exc:
                log.error("scheduled action command to %s failed: %s", addr, exc)
                continue
            for ip, status in results or ():
                if ip and ip != "0.0.0.0":
                    acked[ip] = int(status)
        expected = {c.info.ip_address for c in self.cameras if c.info.ip_address}
        missing = sorted(expected - set(acked))
        bad = {ip: st for ip, st in acked.items() if st != 0}
        if missing or bad:
            parts = []
            if missing:
                parts.append(f"no acknowledgement from {len(missing)}/{n} cameras "
                             f"({', '.join(missing)})")
            if bad:
                parts.append("error status " + ", ".join(
                    f"{ip}=0x{st:04X}" for ip, st in sorted(bad.items())))
            self._ack_problem = ("action command not accepted via "
                                 f"{', '.join(self._addresses)}: " + "; ".join(parts))
            log.error("%s. Set sync.ptp.broadcast_address to the cameras' subnet "
                      "broadcast, or run tools/diagnose_pylon.py.", self._ack_problem)
        else:
            log.info("first action command acknowledged by all %d cameras", n)

    def status(self) -> SyncStatus:
        # Cached from the monitor (every MONITOR_INTERVAL_S): the UI asks several
        # times a second, and reading many cameras over GigE each time adds up.
        per_cam = dict(self._last_status)
        if not per_cam:
            for cam in self.cameras:
                try:
                    per_cam[cam.info.serial] = cam.ptp_status()
                except Exception as exc:
                    per_cam[cam.info.serial] = {"state": f"error: {exc}"}
        offsets = [abs(s["offset_ns"]) for s in per_cam.values()
                   if s.get("state") == "Slave" and isinstance(s.get("offset_ns"), int)]
        worst = max(offsets) if offsets else None
        ready = all(str(s.get("state", "")) in ("Master", "Slave")
                    for s in per_cam.values()) and not self._ack_problem
        detail = f"worst PTP offset {worst} ns" if worst is not None else "PTP active"
        if self._late:
            detail += f"; {self._late} action command errors"
        if self._rejected:
            detail += (f"; cameras rejected {self._rejected} of {self._ack_checks} "
                       f"checked action commands")
        if getattr(self, "_sched_late", 0):
            detail += f"; {self._sched_late} action commands sent too late"
        if self._ack_problem:
            detail += f"; {self._ack_problem}"
        if self._health:
            detail += f"; DEGRADED: {self._health}"
            ready = False
        if self._clock_steps:
            detail += f"; PTP time base jumped {self._clock_steps}x"
        if self._bad_slots:
            detail += f"; {self._bad_slots} frames with implausible timestamps left out"
        return SyncStatus(mode=self.name, ready=ready, detail=detail, per_camera=per_cam)

    def teardown(self) -> None:
        self._stop_monitor.set()
        super().teardown()
        if getattr(self.ptp, "keep_enabled", True):
            return
        for cam in self.cameras:
            try:
                cam.configure_ptp(False)
            except Exception:
                pass


STRATEGIES = {
    "freerun": FreeRunSync,
    "software_trigger": SoftwareTriggerSync,
    "hardware_trigger": HardwareTriggerSync,
    "ptp_action": PtpActionSync,
}


def choose_mode(cfg, cameras: Sequence[CameraBackend]) -> str:
    """Resolve sync.mode, including 'auto'.

    Hardware triggering is never chosen automatically: whether the trigger cable is
    actually plugged in is not something software can detect, and silently picking it
    would produce a recording of nothing.
    """
    mode = cfg.sync.mode
    if mode != "auto":
        return mode
    if not cameras:
        return "freerun"
    if all(c.has(Capability.PTP) and c.has(Capability.ACTION_COMMAND) for c in cameras):
        return "ptp_action"
    if all(c.has(Capability.SOFTWARE_TRIGGER) for c in cameras):
        return "software_trigger"
    return "freerun"


def build_strategy(cfg, cameras: Sequence[CameraBackend]) -> SyncStrategy:
    mode = choose_mode(cfg, cameras)
    cls = STRATEGIES.get(mode)
    if cls is None:
        raise SyncError(f"unknown sync mode {mode!r}")
    return cls(cfg)
