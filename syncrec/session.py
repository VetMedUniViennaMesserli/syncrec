"""Recording session: N cameras -> N ffmpeg processes, paced by one sync strategy.

Each camera gets two threads. The grab thread does nothing but pull frames out of the
driver as fast as they arrive and hand them to a bounded queue; the write thread pipes
them to ffmpeg. Keeping them apart matters: if encoding hiccups, the queue absorbs it
and the camera keeps streaming. If the queue does fill up, frames are dropped
deliberately and counted, which is far better than stalling the driver and losing
frames silently.
"""
from __future__ import annotations

import logging
import copy
import os
import queue
import shutil
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .camera import CameraBackend, Frame
from .config import RecorderConfig
from .sync import SyncStrategy, build_strategy, choose_mode
from .encoders import container_of, resolve as resolve_encoder
from .writer import (FfmpegWriter, MetadataWriter, count_video_frames, find_ffprobe,
                     write_manifest)

log = logging.getLogger(__name__)

CLIP_VARIABLES = {"date": "%Y%m%d", "time": "%H%M%S"}
DEFAULT_CLIP_NAME = "{date}_{time}"
_FORBIDDEN_CHARS = set('\\/:*?"<>|')


def check_clip_name(template: str) -> str:
    """Why `template` cannot be used as a recording name ("" if it can)."""
    bad = sorted({ch for ch in template if ch in _FORBIDDEN_CHARS})
    if bad:
        return f"the name may not contain {' '.join(bad)}"
    try:
        template.format(**{k: "" for k in CLIP_VARIABLES})
    except KeyError as exc:
        return (f"unknown variable {{{exc.args[0]}}}; use "
                + " or ".join(f"{{{k}}}" for k in CLIP_VARIABLES))
    except (ValueError, IndexError):
        return "unmatched { or } in the name"
    if template.strip() in (".", ".."):
        return "the name may not be . or .."
    return ""


def expand_clip_name(template: str, when: Optional[datetime] = None) -> str:
    """Fill in {date} and {time}; blank means DEFAULT_CLIP_NAME."""
    template = (template or "").strip() or DEFAULT_CLIP_NAME
    problem = check_clip_name(template)
    if problem:
        raise ValueError(f"recording name {template!r}: {problem}")
    when = when or datetime.now()
    return template.format(**{k: when.strftime(f) for k, f in CLIP_VARIABLES.items()})


def unique_clip(output_dir: str, clip: str) -> str:
    """`clip`, or clip_2, clip_3, ... if a non-empty folder of that name exists."""
    candidate, n = clip, 1
    while True:
        path = os.path.join(output_dir, candidate)
        if not os.path.isdir(path) or not os.listdir(path):
            return candidate
        n += 1
        candidate = f"{clip}_{n}"


def make_thumbnail(array: np.ndarray) -> np.ndarray:
    """Decimate by striding -- cheap, and the UI only needs a thumbnail."""
    step = max(1, array.shape[1] // 480)
    return np.ascontiguousarray(array[::step, ::step])


@dataclass
class CameraStats:
    serial: str
    grabbed: int = 0
    written: int = 0
    dropped_queue: int = 0       # frames we threw away because encoding fell behind
    placeholders: int = 0        # missed frames filled with the previous image
    duplicate_slots: int = 0     # frames claiming a trigger slot already written
    unslotted: int = 0           # frames whose trigger could not be determined
    skipped_camera: int = 0      # frames the camera itself reports as lost
    timeouts: int = 0
    queue_depth: int = 0
    fps: float = 0.0
    last_host_ns: int = 0
    last_cam_ns: int = 0
    error: str = ""

    def snapshot(self) -> Dict[str, Any]:
        return dict(serial=self.serial, grabbed=self.grabbed, written=self.written,
                    dropped_queue=self.dropped_queue,
                    skipped_camera=self.skipped_camera, timeouts=self.timeouts,
                    placeholders=self.placeholders,
                    duplicate_slots=self.duplicate_slots,
                    unslotted=self.unslotted,
                    queue_depth=self.queue_depth, fps=round(self.fps, 2),
                    error=self.error)


@dataclass
class SyncReport:
    """How far apart the cameras were, measured on frames that share an index."""
    compared: int = 0
    spread_ms_last: float = 0.0
    spread_ms_max: float = 0.0
    spread_ms_mean: float = 0.0
    over_threshold: int = 0
    clock_source: str = "unknown"

    def snapshot(self) -> Dict[str, Any]:
        return dict(compared=self.compared,
                    spread_ms_last=round(self.spread_ms_last, 4),
                    spread_ms_max=round(self.spread_ms_max, 4),
                    spread_ms_mean=round(self.spread_ms_mean, 4),
                    over_threshold=self.over_threshold,
                    clock_source=self.clock_source)


class _SyncMonitor:
    """Groups corresponding frames across cameras and reports their timestamp spread.

    Frames are matched on trigger id where the sync mode provides one. Otherwise the
    per-camera frame counter is used, which is correct as long as no camera has
    dropped a frame -- and dropped frames are counted separately, so a mismatch shows
    up there rather than silently corrupting this measurement.
    """

    def __init__(self, n_cameras: int, threshold_ms: float, keep: int = 256):
        self.n = n_cameras
        self.threshold_ms = threshold_ms
        self.keep = keep
        self._groups: Dict[int, Dict[str, int]] = {}
        self._sum = 0.0
        self._device_clock = 0
        self._host_clock = 0
        # Exposure timestamps are only comparable across cameras on a shared clock
        # (PTP, or simulation); otherwise arrival times are the best there is.
        self.use_device_clock = True
        self.report = SyncReport()
        self._lock = threading.Lock()

    @staticmethod
    def _key(frame: Frame) -> int:
        return frame.trigger_id if frame.trigger_id >= 0 else frame.index

    def add(self, frame: Frame) -> None:
        key = self._key(frame)
        with self._lock:
            # Device timestamps mark exposure; host timestamps mark arrival, which
            # also contains transfer and queueing delay. Which one we had to use
            # decides how the spread figure should be read, so record it.
            if frame.cam_timestamp_ns and self.use_device_clock:
                self._device_clock += 1
                ts = frame.cam_timestamp_ns
            else:
                self._host_clock += 1
                ts = frame.host_timestamp_ns
            group = self._groups.setdefault(key, {})
            group[frame.serial] = ts
            if len(group) < self.n:
                if len(self._groups) > self.keep:
                    # Drop the oldest incomplete group; a camera missed that frame.
                    del self._groups[min(self._groups)]
                return
            spread_ms = (max(group.values()) - min(group.values())) / 1e6
            del self._groups[key]
            r = self.report
            r.compared += 1
            r.spread_ms_last = spread_ms
            r.spread_ms_max = max(r.spread_ms_max, spread_ms)
            self._sum += spread_ms
            r.spread_ms_mean = self._sum / r.compared
            r.clock_source = ("device" if not self._host_clock else
                              "host" if not self._device_clock else "mixed")
            # Arrival times include transfer and queueing, so they are reported but
            # not held against the threshold.
            if spread_ms > self.threshold_ms and r.clock_source == "device":
                r.over_threshold += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return self.report.snapshot()


class RecordingSession:
    """Drives one recording from start to finish.

    Typical use:
        session = RecordingSession(cfg, cameras)
        session.prepare(progress=print)
        session.start()
        ...  # session.stats() / session.preview(serial) while it runs
        session.stop()
    """

    def __init__(self, cfg: RecorderConfig, cameras: Sequence[CameraBackend],
                 clip_name: Optional[str] = None):
        self.cfg = cfg
        self.cameras = list(cameras)
        if not self.cameras:
            raise ValueError("a recording session needs at least one camera")
        root = os.path.abspath(cfg.session.output_dir)
        clip = expand_clip_name(clip_name or cfg.session.clip_name)
        # Never write into an earlier recording (names like "test" get reused).
        self.clip = unique_clip(root, clip)
        if self.clip != clip:
            log.info("a recording named %s already exists; using %s", clip, self.clip)
        self.out_dir = os.path.join(root, self.clip)

        self.strategy: Optional[SyncStrategy] = None
        self.sync_mode = choose_mode(cfg, self.cameras)
        self.monitor = _SyncMonitor(len(self.cameras), cfg.sync.max_spread_ms)

        self.stats: Dict[str, CameraStats] = {c.info.serial: CameraStats(c.info.serial)
                                              for c in self.cameras}
        self._queues: Dict[str, "queue.Queue[Optional[Frame]]"] = {}
        self._writers: Dict[str, FfmpegWriter] = {}
        self._meta: Dict[str, MetadataWriter] = {}
        self._preview: Dict[str, np.ndarray] = {}
        self._preview_lock = threading.Lock()

        self._threads: List[threading.Thread] = []
        self._stop_evt = threading.Event()
        self._stop_lock = threading.Lock()
        self._final_status: Optional[Dict[str, Any]] = None
        self._running = False
        self._prepared = False
        self._t_start = 0.0
        self._t_stop = 0.0
        self.errors: List[str] = []
        self.triggers_issued = 0
        self.triggers_sent = 0
        self._errors_suppressed = 0
        self._verified_segments: set = set()
        self.segment_problems: Dict[int, Dict[str, int]] = {}
        self.trigger_periods_skipped = 0
        self._preview_every = 1

    # ---- setup -------------------------------------------------------------
    def prepare(self, progress: Optional[Callable[[str], None]] = None) -> None:
        """Open and configure cameras, arm the sync strategy, create the writers."""
        say = progress or (lambda _m: None)
        os.makedirs(self.out_dir, exist_ok=True)

        for cam in self.cameras:
            if not cam.is_open():
                cam.open()
            cam.configure(self.cfg)
        say(f"Configured {len(self.cameras)} cameras.")

        self.strategy = build_strategy(self.cfg, self.cameras)
        self.sync_mode = self.strategy.name
        say(f"Sync mode: {self.sync_mode}")
        self._check_disk_space(say)
        self.strategy.prepare(self.cameras, progress=say)
        self.monitor.use_device_clock = self.strategy.shared_device_clock

        # Decide the encoder now (it may test GPU encoders, which takes a second),
        # so a missing GPU is reported before recording rather than mid-way.
        inp, out, desc = resolve_encoder(
            self.cfg.encoder, len(self.cameras),
            container_of(self.cfg.session.filename_pattern))
        self._encoder_cfg = copy.copy(self.cfg.encoder)
        self._encoder_cfg.input_args, self._encoder_cfg.output_args = inp, out
        self.encoder_description = desc
        say(f"Encoder: {desc}")
        log.info("encoder: %s: %s", desc, " ".join(inp + out))
        self._prepared = True

    def _check_disk_space(self, say) -> None:
        free = shutil.disk_usage(self.out_dir).free / 1e9
        say(f"{free:.1f} GB free at {self.out_dir}")
        limit = self.cfg.session.min_free_gb
        if limit > 0 and free < limit:
            raise RuntimeError(f"only {free:.1f} GB free at {self.out_dir}, below "
                               f"session.min_free_gb = {limit:g}")

    # ---- run ---------------------------------------------------------------
    def start(self) -> None:
        if not self._prepared:
            raise RuntimeError("call prepare() before start()")
        if self._running:
            return

        seg = self.segment_frames
        if seg:
            os.makedirs(self._segment_dir(0), exist_ok=True)
            os.makedirs(self._segment_dir(1), exist_ok=True)
            log.info("splitting into segments of %d frames (%.1f min)", seg,
                     seg / self.cfg.acquisition.fps / 60)
        for cam in self.cameras:
            serial = cam.info.serial
            width, height, pix_fmt = cam.output_geometry()
            filename = self.cfg.session.filename_pattern.format(
                clip=self.clip, serial=serial, name=self.cfg.override_for(serial).name
                or serial)
            if seg:
                # ffmpeg expands %04d itself; escape any % already in the path.
                path = os.path.join(self.out_dir.replace("%", "%%"), "segment_%04d",
                                    filename.replace("%", "%%"))
            else:
                path = os.path.join(self.out_dir, filename)
            writer = FfmpegWriter(path, width, height,
                                  self.cfg.acquisition.fps, pix_fmt,
                                  getattr(self, "_encoder_cfg", self.cfg.encoder),
                                  segment_frames=seg)
            writer.open()
            self._writers[serial] = writer
            if self.cfg.encoder.write_metadata_csv:
                meta = MetadataWriter(os.path.join(
                    self._segment_dir(0) if seg else self.out_dir,
                    f"{self.clip}_{serial}.csv"))
                meta.open()
                self._meta[serial] = meta
            self._queues[serial] = queue.Queue(maxsize=self.cfg.acquisition.queue_size)

        self._stop_evt.clear()
        self._running = True
        self._t_start = time.time()
        self._preview_every = max(1, int(self.cfg.acquisition.fps // 10))

        for cam in self.cameras:
            cam.start()
        if self.strategy is not None:
            self.strategy.on_started()

        for cam in self.cameras:
            self._spawn(self._grab_loop, f"grab-{cam.info.serial}", cam)
            self._spawn(self._write_loop, f"write-{cam.info.serial}", cam.info.serial)
        if self.strategy is not None and self.strategy.host_paced:
            self._spawn(self._trigger_loop, "trigger")
        if self.cfg.session.duration_s > 0:
            self._spawn(self._duration_loop, "duration")
        self._spawn(self._housekeeping_loop, "housekeeping")
        self._write_manifest_now()
        log.info("recording %s started (%d cameras, %s)", self.clip, len(self.cameras),
                 self.sync_mode)

    def _spawn(self, target, name, *args) -> None:
        th = threading.Thread(target=target, name=name, args=args, daemon=True)
        th.start()
        self._threads.append(th)

    def _trigger_loop(self) -> None:
        period = 1.0 / self.cfg.acquisition.fps
        base = time.perf_counter()
        trigger_id = 0

        def due(n: int) -> float:
            # The strategy may follow another clock (ptp_action: the cameras' PTP
            # time, which drifts against the host clock by seconds per day).
            t = self.strategy.due_time(n)
            return t if t is not None else base + n * period

        while not self._stop_evt.is_set():
            t = due(trigger_id)
            now = time.perf_counter()
            if t > now:
                if self._stop_evt.wait(t - now):
                    break
            elif now - t > period:
                # The host cannot keep up; skip ahead rather than spiral into a
                # backlog. The trigger id skips too: it fixes the frame's time slot
                # (ptp_action schedules trigger n at t0 + n * period), so the skipped
                # triggers become placeholders and later ones stay on schedule.
                skipped = int((now - t) / period)
                trigger_id += skipped
                self.trigger_periods_skipped += skipped
                log.warning("trigger loop fell behind, skipped %d periods", skipped)
                continue
            try:
                self.strategy.emit_trigger(trigger_id, time.time_ns())
                self.triggers_sent += 1
            except Exception as exc:
                self._fail(f"trigger {trigger_id} failed: {exc}")
            trigger_id += 1
            self.triggers_issued = trigger_id

    def _grab_loop(self, cam: CameraBackend) -> None:
        serial = cam.info.serial
        st = self.stats[serial]
        q = self._queues[serial]
        timeout = self.cfg.acquisition.grab_timeout_ms
        window_start = time.perf_counter()
        window_count = 0
        try:
            while not self._stop_evt.is_set():
                frame = cam.retrieve(timeout)
                if frame is None:
                    st.timeouts += 1
                    if st.grabbed == 0 and st.timeouts == 2:
                        self._fail(
                            f"{serial}: no image received after "
                            f"{2 * timeout / 1000:.0f}s. The camera is not being "
                            f"triggered or its stream is not reaching this PC -- run "
                            f"tools/diagnose_pylon.py to find out which.")
                    continue
                if frame.trigger_id < 0 and self.strategy is not None:
                    slot = self.strategy.slot_for(frame)
                    if slot is not None:
                        frame.trigger_id = slot
                st.grabbed += 1
                st.skipped_camera += frame.skipped_before
                st.last_host_ns = frame.host_timestamp_ns
                st.last_cam_ns = frame.cam_timestamp_ns
                self.monitor.add(frame)

                window_count += 1
                elapsed = time.perf_counter() - window_start
                if elapsed >= 1.0:
                    st.fps = window_count / elapsed
                    window_start, window_count = time.perf_counter(), 0

                if frame.index % self._preview_every == 0:
                    self._store_preview(serial, frame.array)
                try:
                    q.put_nowait(frame)
                    st.queue_depth = q.qsize()
                except queue.Full:
                    st.dropped_queue += 1
                    if self._meta.get(serial):
                        self._meta[serial].write(frame, q.qsize(), written=False)
                    if st.dropped_queue in (1, 10, 100, 1000):
                        log.error("%s: encoder cannot keep up, dropped %d frames",
                                  serial, st.dropped_queue)
        except Exception as exc:
            st.error = str(exc)
            self._fail(f"{serial}: grab loop stopped: {exc}")
        finally:
            try:
                q.put(None, timeout=2.0)          # sentinel: tell the writer to finish
            except queue.Full:
                pass

    def _write_loop(self, serial: str) -> None:
        st = self.stats[serial]
        q = self._queues[serial]
        writer = self._writers[serial]
        meta = self._meta.get(serial)
        fill = self.cfg.session.fill_missing_frames
        slots_expected = bool(self.strategy and self.strategy.provides_slots)
        last_slot = -1
        last_array = None
        seg = self.segment_frames
        vframes = 0                      # frames sent to this camera's video so far
        csv_name = os.path.basename(meta.path) if meta else ""
        # A minute into each segment (or half-way, for short ones) check the last one.
        check_at = min(seg // 2, int(self.cfg.acquisition.fps * 60)) if seg else 0

        def put(array) -> None:
            """Write one video frame; keep the CSV in the same segment as the video."""
            nonlocal vframes
            if seg and vframes and vframes % seg == 0:
                k = vframes // seg
                if meta:
                    meta.switch(os.path.join(self._segment_dir(k), csv_name))
                # ffmpeg opens the next segment's file itself, but does not create
                # its folder: make it a whole segment ahead.
                os.makedirs(self._segment_dir(k + 1), exist_ok=True)
                if serial == next(iter(self._writers)):
                    log.info("segment %d started", k)
            if seg and vframes % seg == check_at and vframes > seg and \
                    serial == next(iter(self._writers)):
                # The previous segment is closed by now: confirm its frame counts.
                k = vframes // seg - 1
                threading.Thread(target=self._verify_segment, args=(k, True, True),
                                 name=f"verify-{k}", daemon=True).start()
            writer.write(array)
            vframes += 1

        while True:
            try:
                frame = q.get(timeout=1.0)
            except queue.Empty:
                if self._stop_evt.is_set():
                    break
                continue
            if frame is None:
                break
            try:
                slot = frame.trigger_id
                if fill and slot < 0 and slots_expected:
                    # No trigger could be assigned (implausible timestamp): placing it
                    # anywhere could shift the video, so leave it out; the slot gets
                    # a placeholder. Kept in the CSV as not written.
                    st.unslotted += 1
                    if meta:
                        meta.write(frame, q.qsize(), written=False)
                    continue
                if fill and slot >= 0:
                    if slot <= last_slot:
                        st.duplicate_slots += 1        # cannot happen with one trigger
                        if meta:                       # per slot; never shift the video
                            meta.write(frame, q.qsize(), written=False)
                        continue
                    for missing in range(last_slot + 1, slot):
                        # Hold the previous image (black before the first one) so
                        # video frame n stays trigger n.
                        filler = (last_array if last_array is not None
                                  else np.zeros_like(frame.array))
                        put(filler)
                        st.placeholders += 1
                        if meta:
                            meta.write_placeholder(missing)
                    last_slot = slot
                    last_array = frame.array
                put(frame.array)
                st.written += 1
                if meta:
                    meta.write(frame, q.qsize(), written=True)
            except Exception as exc:
                st.error = str(exc)
                self._fail(f"{serial}: write failed: {exc}")
                break

    @property
    def segment_frames(self) -> int:
        minutes = self.cfg.session.segment_minutes or 0.0
        return int(round(minutes * 60 * self.cfg.acquisition.fps)) if minutes > 0 else 0

    def _segment_dir(self, k: int) -> str:
        return os.path.join(self.out_dir, f"segment_{k:04d}")

    def _verify_segment(self, k: int, full: bool, live: bool = False) -> None:
        """Check that every camera's file in segment k has exactly the frames it should."""
        if k in self._verified_segments:
            return
        self._verified_segments.add(k)
        ffprobe = find_ffprobe(self.cfg.encoder.ffmpeg)
        if not ffprobe:
            return
        folder = self._segment_dir(k)
        if not os.path.isdir(folder):
            return
        counts = {}
        for name in sorted(os.listdir(folder)):
            if os.path.splitext(name)[1].lower() in (".mp4", ".mkv", ".mov", ".avi", ".mxf"):
                counts[name] = count_video_frames(ffprobe, os.path.join(folder, name))
        if not counts:
            return
        if live and any(c < 0 for c in counts.values()):
            # Not finalised yet (a slow encoder still holds it); retry at the end.
            self._verified_segments.discard(k)
            return
        expected = self.segment_frames
        bad = {n: c for n, c in counts.items() if (c != expected if full else c > expected)}
        if bad:
            self._fail(f"segment {k}: frame counts {bad}, expected {expected}")
            self.segment_problems[k] = bad
        else:
            log.info("segment %d verified: %d frames in each of %d files",
                     k, expected if full else max(counts.values()), len(counts))

    def _housekeeping_loop(self) -> None:
        """Every 30 s: disk space. Every 5 min: refresh manifest.json.

        Stops cleanly before the disk is full (a full disk corrupts open files). The
        manifest is kept current during the recording, so even after a crash or power
        cut the recording folder says what was recorded, how, and how well.
        """
        limit = self.cfg.session.min_free_gb
        ticks = 0
        while not self._stop_evt.wait(30.0):
            ticks += 1
            if ticks % 10 == 0:
                self._write_manifest_now()
            if limit <= 0:
                continue
            try:
                free = shutil.disk_usage(self.out_dir).free / 1e9
            except OSError:
                continue
            if free < limit:
                self._fail(f"only {free:.1f} GB free at {self.out_dir} (limit "
                           f"session.min_free_gb = {limit:g}); stopping the recording")
                threading.Thread(target=self.stop, name="disk-stop", daemon=True).start()
                return

    def _write_manifest_now(self) -> None:
        if self._final_status is not None:
            return                                   # the final one is written
        try:
            status = self.status()
            status.update(self._static_manifest())
            write_manifest(os.path.join(self.out_dir, "manifest.json"), status)
        except Exception as exc:
            log.debug("manifest update failed: %s", exc)

    def _static_manifest(self) -> Dict[str, Any]:
        enc = getattr(self, "_encoder_cfg", self.cfg.encoder)
        return {
            "config": self.cfg.to_dict(),
            "cameras_info": {c.info.serial: vars(c.info) for c in self.cameras},
            "encoder": {"description": getattr(self, "encoder_description", ""),
                        "input_args": enc.input_args, "output_args": enc.output_args},
            "started_at": datetime.fromtimestamp(self._t_start).isoformat()
            if self._t_start else None,
        }

    def _duration_loop(self) -> None:
        if self._stop_evt.wait(self.cfg.session.duration_s):
            return
        log.info("reached configured duration of %.1fs", self.cfg.session.duration_s)
        threading.Thread(target=self.stop, name="auto-stop", daemon=True).start()

    def _fail(self, message: str) -> None:
        log.error("%s", message)
        # Bounded: a recording can run for weeks.
        if len(self.errors) < 500:
            self.errors.append(message)
        else:
            self._errors_suppressed += 1

    def _store_preview(self, serial: str, array: np.ndarray) -> None:
        thumb = make_thumbnail(array)
        with self._preview_lock:
            self._preview[serial] = thumb

    # ---- inspection --------------------------------------------------------
    def preview(self, serial: str) -> Optional[np.ndarray]:
        with self._preview_lock:
            return self._preview.get(serial)

    def is_running(self) -> bool:
        return self._running

    def elapsed(self) -> float:
        if not self._t_start:
            return 0.0
        return (self._t_stop or time.time()) - self._t_start

    def status(self) -> Dict[str, Any]:
        for serial, q in self._queues.items():
            self.stats[serial].queue_depth = q.qsize()
        now = time.time_ns()
        for st in self.stats.values():
            # fps is measured per arriving frame; a camera that stopped delivering
            # would otherwise keep showing its last rate.
            if self._running and st.last_host_ns and now - st.last_host_ns > 2e9:
                st.fps = 0.0
        sync_status = self.strategy.status() if self.strategy else None
        total_bytes = sum(w.bytes_written for w in self._writers.values())
        elapsed = max(1e-6, self.elapsed())
        return {
            "clip": self.clip,
            "output_dir": self.out_dir,
            "running": self._running,
            "elapsed_s": round(elapsed, 2),
            "sync_mode": self.sync_mode,
            "sync_ready": bool(sync_status.ready) if sync_status else False,
            "sync_detail": sync_status.detail if sync_status else "",
            "sync_per_camera": sync_status.per_camera if sync_status else {},
            "triggers_issued": self.triggers_issued,
            "triggers_sent": self.triggers_sent,
            "trigger_periods_skipped": self.trigger_periods_skipped,
            "cameras": {s: st.snapshot() for s, st in self.stats.items()},
            "sync_report": self.monitor.snapshot(),
            "throughput_mb_s": round(total_bytes / 1e6 / elapsed, 1),
            "errors": list(self.errors) + (
                [f"... and {self._errors_suppressed} more errors (see the log)"]
                if self._errors_suppressed else []),
        }

    # ---- shutdown ----------------------------------------------------------
    def stop(self, timeout: float = 30.0) -> Dict[str, Any]:
        """Stop triggering, let the queues drain, close everything, write a manifest.

        Idempotent and safe to call from several threads: the duration timer and the
        operator often race, and the loser must wait for the real shutdown to finish
        rather than return a half-written status.
        """
        with self._stop_lock:
            if self._final_status is not None:
                return self._final_status
            if not self._running:
                return self.status()
            return self._stop_locked(timeout)

    def _stop_locked(self, timeout: float) -> Dict[str, Any]:
        log.info("stopping recording %s", self.clip)
        self._running = False
        self._t_stop = time.time()
        self._stop_evt.set()

        # pylon's own transport counters (resends, incomplete buffers, ...). Read them
        # before stopping: they are what tells network loss apart from host problems.
        stream_stats = {}
        for cam in self.cameras:
            getter = getattr(cam, "statistics", None)
            if getter is not None:
                try:
                    stream_stats[cam.info.serial] = getter()
                except Exception:
                    pass

        for cam in self.cameras:
            try:
                cam.stop()
            except Exception as exc:
                self._fail(f"{cam.info.serial}: stop failed: {exc}")

        deadline = time.time() + timeout
        for th in self._threads:
            th.join(timeout=max(0.1, deadline - time.time()))
        self._threads.clear()

        for serial, writer in self._writers.items():
            err = writer.close()
            if err:
                self._fail(err)
        for meta in self._meta.values():
            meta.close()
        self.segments = []
        if self.segment_frames:
            # Verify what was not verified during the recording; the last segment
            # may be shorter.
            folders = sorted(n for n in os.listdir(self.out_dir) if n.startswith("segment_"))
            last = max((int(n.split("_")[1]) for n in folders
                        if os.listdir(os.path.join(self.out_dir, n))), default=-1)
            for k in range(last + 1):
                self._verify_segment(k, full=k < last)
            # The folder made ahead for the next segment stays empty; remove it.
            for name in sorted(os.listdir(self.out_dir)):
                path = os.path.join(self.out_dir, name)
                if name.startswith("segment_") and os.path.isdir(path):
                    if os.listdir(path):
                        self.segments.append(name)
                    else:
                        os.rmdir(path)

        if self.strategy is not None:
            try:
                self.strategy.teardown()
            except Exception as exc:
                self._fail(f"sync teardown failed: {exc}")

        status = self.status()
        status["running"] = False
        status["files"] = {s: w.path for s, w in self._writers.items()}
        status["frames_written"] = {s: w.frames_written for s, w in self._writers.items()}
        if self.segment_frames:
            status["segments"] = {"frames_per_segment": self.segment_frames,
                                  "folders": getattr(self, "segments", []),
                                  "problems": {str(k): v for k, v in
                                               self.segment_problems.items()}}
        status["stream_statistics"] = stream_stats
        dur = max(1e-6, self._t_stop - self._t_start)
        status["average_fps"] = {s: round(st.grabbed / dur, 2) for s, st in self.stats.items()}
        status["trigger_rate"] = round(self.triggers_sent / dur, 2)
        status.update(self._static_manifest())
        status["stopped_at"] = datetime.fromtimestamp(self._t_stop).isoformat()
        write_manifest(os.path.join(self.out_dir, "manifest.json"), status)
        log.info("recording %s finished: %s", self.clip,
                 {s: w.frames_written for s, w in self._writers.items()})
        self._final_status = status
        return status
