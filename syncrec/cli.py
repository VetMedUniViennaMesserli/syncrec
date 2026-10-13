"""Headless entry point: syncrec record / list / config / gui / view.

The GUI is the normal way to drive a recording, but everything it does is available
here too, which is what you want for unattended or scripted runs.
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from typing import List, Optional

from .backends import discover
from .config import DEFAULT_CONFIG_PATH, RecorderConfig
from .session import RecordingSession

log = logging.getLogger("syncrec")


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S")


def load_config(path: Optional[str]) -> RecorderConfig:
    if path:
        cfg = RecorderConfig.load(path)
        log.info("loaded config from %s", path)
        return cfg
    if os.path.isfile(DEFAULT_CONFIG_PATH):
        log.info("loaded config from %s", DEFAULT_CONFIG_PATH)
        return RecorderConfig.load(DEFAULT_CONFIG_PATH)
    log.info("no config file given and no %s, using built-in defaults",
             DEFAULT_CONFIG_PATH)
    return RecorderConfig()


def apply_overrides(cfg: RecorderConfig, args) -> RecorderConfig:
    if args.backend:
        cfg.acquisition.backend = args.backend
    if args.sync:
        cfg.sync.mode = args.sync
    if args.fps:
        cfg.acquisition.fps = args.fps
    if args.duration is not None:
        cfg.session.duration_s = args.duration
    if args.output:
        cfg.session.output_dir = args.output
    if args.segment_minutes is not None:
        cfg.session.segment_minutes = args.segment_minutes
    if args.clip:
        cfg.session.clip_name = args.clip
    if args.codec:
        cfg.encoder.output_args = args.codec.split()
        cfg.encoder.codec = "custom"
    if args.video_codec:
        cfg.encoder.codec = args.video_codec
    if args.hw:
        cfg.encoder.hardware = args.hw
    if args.quality is not None:
        cfg.encoder.quality = args.quality
    cfg.validate()
    return cfg


def cmd_list(args) -> int:
    cfg = apply_overrides(load_config(args.config), args)
    cams, backend = discover(cfg)
    print(f"backend: {backend}   cameras: {len(cams)}")
    for cam in cams:
        try:
            cam.open()
            caps = ", ".join(sorted(c.value for c in cam.capabilities())) or "-"
            geom = "?"
            try:
                cam.configure(cfg)
                w, h, fmt = cam.output_geometry()
                geom = f"{w}x{h} {fmt}"
            except Exception as exc:
                geom = f"(configure failed: {exc})"
            ptp = cam.ptp_status().get("state", "-")
            print(f"  {cam.info.serial:<20} {cam.info.model:<16} "
                  f"{cam.info.device_class:<14} {geom:<18} ptp={ptp}")
            print(f"      caps: {caps}")
        finally:
            cam.close()
    return 0


def cmd_config(args) -> int:
    cfg = load_config(args.config) if args.config else RecorderConfig()
    path = args.write or "config.json"
    cfg.save(path)
    print(f"wrote {os.path.abspath(path)}")
    return 0


def cmd_record(args) -> int:
    cfg = apply_overrides(load_config(args.config), args)
    cams, backend = discover(cfg)
    if not cams:
        log.error("no cameras available")
        return 2
    log.info("using %s backend with %d cameras: %s", backend, len(cams),
             ", ".join(c.info.serial for c in cams))

    session = RecordingSession(cfg, cams)
    stop_evt = threading.Event()

    def handle_signal(_sig, _frm):
        if not stop_evt.is_set():
            log.info("stop requested")
            stop_evt.set()

    signal.signal(signal.SIGINT, handle_signal)
    try:
        signal.signal(signal.SIGTERM, handle_signal)
    except (AttributeError, ValueError):
        pass

    try:
        session.prepare(progress=lambda m: log.info("%s", m))
        session.start()
    except Exception as exc:
        log.error("could not start: %s", exc)
        try:
            session.stop()
        except Exception:
            pass
        for cam in cams:
            cam.close()
        return 1

    print(f"recording to {session.out_dir} -- press Ctrl+C to stop")
    try:
        while session.is_running() and not stop_evt.is_set():
            time.sleep(1.0)
            _print_status_line(session)
    finally:
        print()
        status = session.stop()
        for cam in cams:
            cam.close()

    _print_summary(status)
    return 1 if status.get("errors") else 0


def _print_status_line(session: RecordingSession) -> None:
    st = session.status()
    cams = st["cameras"].values()
    grabbed = sum(c["grabbed"] for c in cams)
    dropped = sum(c["dropped_queue"] for c in cams)
    fps = sum(c["fps"] for c in cams) / max(1, len(st["cameras"]))
    spread = st["sync_report"]["spread_ms_last"]
    sys.stdout.write(
        f"\r{st['elapsed_s']:7.1f}s  {grabbed:7d} frames  {fps:5.1f} fps/cam  "
        f"spread {spread:7.3f} ms  dropped {dropped:5d}  {st['throughput_mb_s']:6.1f} MB/s ")
    sys.stdout.flush()


def _print_summary(status) -> None:
    print(f"\nclip     : {status['clip']}")
    print(f"output   : {status['output_dir']}")
    print(f"duration : {status['elapsed_s']:.1f} s   sync: {status['sync_mode']} "
          f"({status['sync_detail']})")
    rep = status["sync_report"]
    print(f"sync     : compared {rep['compared']} frame groups, "
          f"mean {rep['spread_ms_mean']:.3f} ms, max {rep['spread_ms_max']:.3f} ms, "
          f"{rep['over_threshold']} over threshold")
    if rep.get("clock_source") != "device":
        print("           (measured from host arrival times: without PTP the cameras'")
        print("            own timestamps are not on a shared clock, so this includes")
        print("            transfer delay and overstates the true exposure spread)")
    print(f"{'camera':<20}{'grabbed':>9}{'written':>9}{'dropped':>9}{'missed':>9}"
          f"{'filled':>9}")
    for serial, c in sorted(status["cameras"].items()):
        print(f"{serial:<20}{c['grabbed']:>9}{c['written']:>9}{c['dropped_queue']:>9}"
              f"{c['skipped_camera']:>9}{c.get('placeholders', 0):>9}")
        if c["error"]:
            print(f"    error: {c['error']}")
    for err in status.get("errors", []):
        print(f"ERROR: {err}")


def cmd_gui(args) -> int:
    from .ui.app import run_gui
    return run_gui(args.config)


def cmd_encoders(args) -> int:
    """Test which codec / hardware combinations work on this PC."""
    from .encoders import CODECS, HARDWARE, survey
    cfg = apply_overrides(load_config(args.config), args)
    n = args.sessions
    print(f"Testing encoders with {cfg.encoder.ffmpeg} "
          f"({n} parallel encodes per GPU, one per camera)...\n")
    for codec, hw, ok, msg in survey(cfg.encoder.ffmpeg, n):
        print(f"  [{' OK ' if ok else ' -- '}] {CODECS[codec]:<28} {HARDWARE[hw]:<32} {msg}")
    return 0


def cmd_check(args) -> int:
    """Inspect the PC and cameras; recommend settings (tools/system_check.py)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    from tools.system_check import main as run_check
    return run_check(args.check_args)


def cmd_view(args) -> int:
    from .ui.viewer import main as run_viewer
    return run_viewer([args.folder] if args.folder else [])


def build_parser() -> argparse.ArgumentParser:
    # Shared options are attached to the top level *and* to every subcommand, so
    # `syncrec --fps 30 record` and `syncrec record --fps 30` both work.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-c", "--config",
                        help="path to a config JSON file (default: settings.json)")
    common.add_argument("-v", "--verbose", action="store_true")
    common.add_argument("--backend", choices=["auto", "pylon", "dummy"])
    common.add_argument("--sync", choices=["auto", "ptp_action", "hardware_trigger",
                                           "software_trigger", "freerun"])
    common.add_argument("--fps", type=float)
    common.add_argument("--duration", type=float, help="seconds; 0 = until stopped")
    common.add_argument("--output", help="output directory")
    common.add_argument("--segment-minutes", type=float,
                        help="split into segments of this many minutes (0 = off)")
    common.add_argument("--clip", help="clip name (default: timestamp)")
    # The value starts with a dash, so argparse needs the = form:
    #   --codec="-c:v h264_nvenc -cq 20"
    common.add_argument("--video-codec", choices=["h264", "h265", "av1", "mjpeg", "ffv1"],
                        help="video codec (ffmpeg arguments are generated)")
    common.add_argument("--hw", choices=["auto", "cpu", "nvidia", "intel", "amd", "vaapi"],
                        help="where to encode: auto = GPU if one works, else CPU")
    common.add_argument("--quality", type=int,
                        help="encoder quality, lower = better (default per codec)")
    common.add_argument("--codec",
                        help='ffmpeg output args; must be written with an equals '
                             'sign, e.g. --codec="-c:v h264_nvenc -cq 20"')

    p = argparse.ArgumentParser(
        prog="syncrec", parents=[common],
        description="Synchronised multi-camera video recorder for Basler cameras.")
    sub = p.add_subparsers(dest="command")
    sub.add_parser("list", parents=[common],
                   help="list the cameras that would be used")
    sub.add_parser("record", parents=[common],
                   help="record until Ctrl+C or --duration")
    cfg_p = sub.add_parser("config", parents=[common],
                           help="write a config file with the defaults")
    cfg_p.add_argument("--write", help="path to write (default: config.json)")
    sub.add_parser("gui", parents=[common], help="launch the recorder UI")
    enc_p = sub.add_parser("encoders", parents=[common],
                           help="test which codecs / GPU encoders work on this PC")
    enc_p.add_argument("--sessions", type=int, default=8,
                       help="parallel encodes to test per GPU (= cameras, default 8)")
    sub.add_parser(
        "check", add_help=False,
        help="inspect this PC and the cameras, recommend settings (check --help)")
    view_p = sub.add_parser("view", help="play a recording's videos frame by frame")
    view_p.add_argument("folder", nargs="?", help="recording folder (default: choose)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "check":
        # Hand every option to the check tool unparsed (it has its own --help).
        return cmd_check(argparse.Namespace(check_args=argv[1:]))
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    handlers = {"list": cmd_list, "record": cmd_record, "config": cmd_config,
                "gui": cmd_gui, "view": cmd_view,
                "encoders": cmd_encoders, "check": cmd_check}
    handler = handlers.get(args.command or "gui")
    try:
        return handler(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        log.error("%s", exc, exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
