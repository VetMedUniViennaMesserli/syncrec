"""End-to-end self test: record with dummy cameras in every sync mode and verify.

Run this after changing anything in the recorder, and once on a new recording
machine; it needs no cameras. It proves the whole chain works -- discovery, arming,
triggering, encoding, metadata, verification -- and that the synchronised modes really
do produce synchronised output while free run does not.

    python tools/selftest.py
    python tools/selftest.py --keep --out selftest_output
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from syncrec.backends import discover                       # noqa: E402
from syncrec.config import RecorderConfig                   # noqa: E402
from syncrec.session import RecordingSession                # noqa: E402
from tools.check_sync import analyse, load_metadata        # noqa: E402

# Free run is expected to be out of sync -- that is the point of the comparison.
EXPECTATIONS = {
    "ptp_action":       {"max_spread_ms": 1.0},
    "hardware_trigger": {"max_spread_ms": 1.0},
    "software_trigger": {"max_spread_ms": 100.0},
    "freerun":          {"min_spread_ms": 1.0},
}


def run_one(mode: str, out_dir: str, seconds: float, n_cams: int,
            width: int, height: int, fps: float) -> dict:
    cfg = RecorderConfig.from_dict({
        "session": {"output_dir": out_dir, "duration_s": seconds},
        "acquisition": {"backend": "dummy", "fps": fps, "queue_size": 200,
                        "roi": {"width": width, "height": height}},
        "sync": {"mode": mode, "ptp": {"sync_timeout_s": 20.0}},
        "encoder": {"output_args": ["-c:v", "libx264", "-preset", "ultrafast",
                                    "-crf", "28", "-pix_fmt", "yuv420p"]},
        "dummy": {"count": n_cams},
    })
    cams, _backend = discover(cfg)
    session = RecordingSession(cfg, cams, clip_name=mode)
    try:
        session.prepare()
        session.start()
        while session.is_running():
            session.status()
            import time
            time.sleep(0.2)
    finally:
        status = session.stop()
        for cam in cams:
            cam.close()
    report = analyse(load_metadata(session.out_dir), tolerance_ms=1.0)
    return {"status": status, "report": report, "dir": session.out_dir}


def check(mode: str, result: dict, n_cams: int) -> list:
    """Return a list of failure strings; empty means the mode behaved as expected."""
    failures = []
    report, status = result["report"], result["status"]
    expect = EXPECTATIONS[mode]
    spread = report["spread_ms"]["max"]

    if len(report["serials"]) != n_cams:
        failures.append(f"expected {n_cams} cameras in the metadata, "
                        f"got {len(report['serials'])}")
    counts = set(report["counts"].values())
    if not counts or max(counts) < 5:
        failures.append(f"almost no frames were recorded: {sorted(counts)}")
    if "max_spread_ms" in expect and spread > expect["max_spread_ms"]:
        failures.append(f"spread {spread:.3f} ms exceeds the "
                        f"{expect['max_spread_ms']} ms this mode promises")
    if "min_spread_ms" in expect and spread < expect["min_spread_ms"]:
        failures.append(f"spread {spread:.3f} ms is suspiciously good for free run "
                        f"-- the dummy cameras are not drifting as they should")
    for serial, cam in status["cameras"].items():
        if cam["error"]:
            failures.append(f"{serial}: {cam['error']}")
        if cam["dropped_queue"]:
            failures.append(f"{serial}: dropped {cam['dropped_queue']} frames")
    for err in status.get("errors", []):
        failures.append(err)
    for serial, path in status.get("files", {}).items():
        if not os.path.isfile(path) or os.path.getsize(path) < 1000:
            failures.append(f"{serial}: {path} is missing or empty")
    return failures


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", help="where to record (default: a temp folder)")
    parser.add_argument("--keep", action="store_true", help="keep the recordings")
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--cameras", type=int, default=8)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    out_dir = args.out or tempfile.mkdtemp(prefix="syncrec_selftest_")
    print(f"recording {args.cameras} dummy cameras at {args.width}x{args.height}@"
          f"{args.fps:g} for {args.seconds:g}s per mode")
    print(f"output: {out_dir}\n")

    all_failures = {}
    try:
        for mode in EXPECTATIONS:
            print(f"--- {mode}")
            try:
                result = run_one(mode, out_dir, args.seconds, args.cameras,
                                 args.width, args.height, args.fps)
            except Exception as exc:
                all_failures[mode] = [f"raised {type(exc).__name__}: {exc}"]
                print(f"    FAILED to run: {exc}")
                if args.verbose:
                    raise
                continue
            report = result["report"]
            counts = sorted(set(report["counts"].values()))
            print(f"    frames {counts}  groups {report['complete_groups']}"
                  f"/{report['total_groups']}  "
                  f"spread mean {report['spread_ms']['mean']:.4f} ms / "
                  f"max {report['spread_ms']['max']:.4f} ms")
            failures = check(mode, result, args.cameras)
            all_failures[mode] = failures
            for failure in failures:
                print(f"    FAIL: {failure}")
            if not failures:
                print("    ok")
    finally:
        if not args.keep and not args.out:
            shutil.rmtree(out_dir, ignore_errors=True)
        elif args.keep:
            print(f"\nrecordings kept in {out_dir}")

    failed = {m: f for m, f in all_failures.items() if f}
    print()
    if failed:
        print(f"SELF TEST FAILED: {', '.join(failed)}")
        return 1
    print(f"SELF TEST PASSED ({len(all_failures)} sync modes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
