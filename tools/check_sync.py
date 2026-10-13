"""Verify a finished recording: frame counts, gaps, and cross-camera sync.

Run this after every session that matters. It reads the per-frame CSVs the recorder
writes next to the videos and answers the question that counts: did every camera
capture the same instants?

    python tools/check_sync.py recordings/20260925_143000
    python tools/check_sync.py recordings/20260925_143000 --tolerance-ms 0.5 --verbose
    python tools/check_sync.py recordings/20260925_143000 --segment 3

Split recordings (segment_0000/, segment_0001/, ...) are checked as a whole, one
segment at a time so that recordings of days or weeks fit in memory, with a line
per segment; --segment N checks just one. A segment folder can also be given
directly.

Two separate verdicts:
  SYNC          frames that every camera recorded were exposed within the tolerance
  COMPLETENESS  how many triggers a camera has no real image for (the video holds a
                placeholder there, so the videos stay aligned, but the image is missing)
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
from array import array
from typing import Dict, List, Optional, Tuple

VIDEO_EXT = (".mp4", ".mkv", ".avi", ".mov")


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
def segment_dirs(directory: str) -> List[str]:
    """The folders holding CSVs: the segments of a split recording, else itself."""
    segs = sorted(d for d in glob.glob(os.path.join(directory, "segment_*"))
                  if os.path.isdir(d))
    return segs or [directory]


def load_metadata(directory: str, extra: Optional[Dict[str, Dict[str, int]]] = None
                  ) -> Dict[str, List[dict]]:
    """Real frames from every <clip>_<serial>.csv in one folder, keyed by serial.

    Placeholders and frames dropped before encoding are not real images and are
    left out; with `extra`, they are counted there per serial instead. A recording
    root with segment folders is read segment after segment (fine for short ones;
    long ones are better checked with check_recording, which streams).
    """
    data: Dict[str, List[dict]] = {}
    for folder in segment_dirs(directory):
        for path in sorted(glob.glob(os.path.join(folder, "*.csv"))):
            serial = os.path.splitext(os.path.basename(path))[0].rsplit("_", 1)[-1]
            counts = extra.setdefault(serial, {"placeholders": 0, "not_written": 0}) \
                if extra is not None else None
            rows = data.setdefault(serial, [])
            with open(path, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    if row.get("written") == "0":
                        if counts is not None:
                            counts["not_written"] += 1
                        continue
                    if row.get("placeholder") == "1":
                        if counts is not None:
                            counts["placeholders"] += 1
                        continue
                    rows.append({
                        "frame_index": int(row["frame_index"]),
                        "block_id": int(row["block_id"]) if row["block_id"] else None,
                        "trigger_id": int(row["trigger_id"]),
                        "cam_ns": int(row["cam_timestamp_ns"]),
                        "host_ns": int(row["host_timestamp_ns"]),
                        "skipped_before": int(row["skipped_before"]),
                    })
    return {k: v for k, v in data.items() if v}


# ---------------------------------------------------------------------------
# analysis of one block of frames (one segment, or a whole short recording)
# ---------------------------------------------------------------------------
def frame_key(row: dict) -> int:
    return row["trigger_id"] if row["trigger_id"] >= 0 else row["frame_index"]


def analyse(data: Dict[str, List[dict]], tolerance_ms: float,
            device_clock: bool = True, keep_spreads: bool = False) -> dict:
    """device_clock: compare exposure timestamps (only meaningful when the cameras
    share one clock, i.e. PTP); otherwise host arrival times are compared."""
    def timestamp(row: dict) -> int:
        return (row["cam_ns"] if device_clock else 0) or row["host_ns"]

    serials = sorted(data)
    by_key: Dict[int, Dict[str, int]] = {}
    for serial, rows in data.items():
        for row in rows:
            by_key.setdefault(frame_key(row), {})[serial] = timestamp(row)

    spreads = sorted((max(v.values()) - min(v.values())) / 1e6
                     for v in by_key.values() if len(v) == len(serials))
    incomplete = {k: sorted(set(serials) - set(v)) for k, v in by_key.items()
                  if len(v) < len(serials)}

    real = [r for rows in data.values() for r in rows]
    device = sum(1 for r in real if r["cam_ns"] and device_clock)
    clock_source = ("device" if device == len(real) else
                    "host" if device == 0 else "mixed")

    # Frame interval and its jitter, from consecutive triggers only: a missing
    # frame is a completeness question, not jitter.
    intervals: Dict[str, Tuple[float, float, int]] = {}
    gaps: Dict[str, int] = {}
    for serial, rows in data.items():
        deltas = [(timestamp(b) - timestamp(a)) / 1e6 for a, b in zip(rows, rows[1:])
                  if frame_key(b) - frame_key(a) == 1]
        if deltas:
            mean = sum(deltas) / len(deltas)
            intervals[serial] = (mean, max(abs(d - mean) for d in deltas), len(deltas))
        gaps[serial] = sum(r["skipped_before"] for r in rows)

    keys = [frame_key(r) for r in real]
    report = {
        "serials": serials,
        "counts": {s: len(rows) for s, rows in data.items()},
        "complete_groups": len(spreads),
        "total_groups": len(by_key),
        "first_key": min(keys) if keys else 0,
        "last_key": max(keys) if keys else -1,
        "incomplete": incomplete,
        "spread_ms": {
            "mean": sum(spreads) / len(spreads) if spreads else 0.0,
            "median": spreads[len(spreads) // 2] if spreads else 0.0,
            "p95": spreads[int(len(spreads) * 0.95)] if spreads else 0.0,
            "max": spreads[-1] if spreads else 0.0,
        },
        "over_tolerance": sum(1 for s in spreads if s > tolerance_ms),
        "clock_source": clock_source,
        "intervals": {s: (m, w) for s, (m, w, _n) in intervals.items()},
        "_intervals_n": intervals,
        "camera_gaps": gaps,
    }
    if keep_spreads:
        report["_spreads"] = spreads
    return report


# ---------------------------------------------------------------------------
# a whole recording, segment by segment
# ---------------------------------------------------------------------------
def check_recording(directory: str, tolerance_ms: float, device_clock: bool,
                    only_segment: Optional[int] = None, progress=None) -> dict:
    folders = segment_dirs(directory)
    if only_segment is not None:
        folders = [f for f in folders
                   if os.path.basename(f) == f"segment_{only_segment:04d}"]
    total: dict = {"serials": set(), "counts": {}, "placeholders": {}, "not_written": {},
                   "camera_gaps": {}, "complete_groups": 0, "total_groups": 0,
                   "over_tolerance": 0, "incomplete": {}, "incomplete_count": 0,
                   "clock_sources": set(), "segments": [], "last_key": -1,
                   "first_key": None}
    spreads = array("f")
    interval_acc: Dict[str, List[float]] = {}        # sum, n, worst
    for folder in folders:
        extra: Dict[str, Dict[str, int]] = {}
        data = load_metadata(folder, extra)
        if not data:
            continue
        rep = analyse(data, tolerance_ms, device_clock, keep_spreads=True)
        name = os.path.basename(folder) if folder != directory else "(recording)"
        spreads.extend(rep["_spreads"])
        total["serials"].update(rep["serials"])
        for s in set(rep["counts"]) | set(extra):
            total["counts"][s] = total["counts"].get(s, 0) + rep["counts"].get(s, 0)
            e = extra.get(s, {})
            total["placeholders"][s] = total["placeholders"].get(s, 0) + e.get("placeholders", 0)
            total["not_written"][s] = total["not_written"].get(s, 0) + e.get("not_written", 0)
            total["camera_gaps"][s] = total["camera_gaps"].get(s, 0) + \
                rep["camera_gaps"].get(s, 0)
        for s, (mean, worst, n) in rep["_intervals_n"].items():
            acc = interval_acc.setdefault(s, [0.0, 0, 0.0])
            acc[0] += mean * n
            acc[1] += n
            acc[2] = max(acc[2], worst)
        total["complete_groups"] += rep["complete_groups"]
        total["total_groups"] += rep["total_groups"]
        total["over_tolerance"] += rep["over_tolerance"]
        total["incomplete_count"] += len(rep["incomplete"])
        total["last_key"] = max(total["last_key"], rep["last_key"])
        if rep["total_groups"]:
            total["first_key"] = rep["first_key"] if total["first_key"] is None \
                else min(total["first_key"], rep["first_key"])
        for k, v in rep["incomplete"].items():
            if len(total["incomplete"]) < 2000:
                total["incomplete"][k] = v
        total["clock_sources"].add(rep["clock_source"])
        total["segments"].append({
            "name": name, "frames": rep["counts"], "groups": rep["total_groups"],
            "complete": rep["complete_groups"], "max_ms": rep["spread_ms"]["max"],
            "over": rep["over_tolerance"],
            "placeholders": sum(e.get("placeholders", 0) for e in extra.values()),
        })
        if progress:
            progress(name)
        del data, rep
    total["serials"] = sorted(total["serials"])
    srt = sorted(spreads)
    n = len(srt)
    total["spread_ms"] = {"mean": sum(srt) / n if n else 0.0,
                          "median": srt[n // 2] if n else 0.0,
                          "p95": srt[int(n * 0.95)] if n else 0.0,
                          "max": srt[-1] if n else 0.0}
    total["intervals"] = {s: (a[0] / a[1] if a[1] else 0.0, a[2])
                          for s, a in interval_acc.items()}
    cs = total.pop("clock_sources")
    total["clock_source"] = cs.pop() if len(cs) == 1 else "mixed"
    return total


def probe_videos(directory: str, only_segment: Optional[int] = None) -> Dict[str, dict]:
    """Ask ffprobe what actually landed in the video files, summed over segments."""
    result: Dict[str, dict] = {}
    folders = segment_dirs(directory)
    if only_segment is not None:
        folders = [f for f in folders
                   if os.path.basename(f) == f"segment_{only_segment:04d}"]
    for folder in folders:
        for path in sorted(p for p in glob.glob(os.path.join(folder, "*"))
                           if p.lower().endswith(VIDEO_EXT)):
            serial = os.path.splitext(os.path.basename(path))[0].rsplit("_", 1)[-1]
            entry = result.setdefault(serial, {"frames": 0, "bytes": 0, "files": 0})
            try:
                out = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "v:0",
                     "-count_packets", "-show_entries",
                     "stream=nb_read_packets,width,height,r_frame_rate,codec_name",
                     "-of", "json", path],
                    capture_output=True, text=True, timeout=600)
                stream = json.loads(out.stdout)["streams"][0]
                entry["frames"] += int(stream.get("nb_read_packets", 0))
                entry["bytes"] += os.path.getsize(path)
                entry["files"] += 1
                entry["size"] = f"{stream['width']}x{stream['height']}"
                entry["codec"] = stream["codec_name"]
            except Exception as exc:
                entry.setdefault("errors", []).append(f"{path}: {exc}")
    return result


# ---------------------------------------------------------------------------
def find_manifest(directory: str) -> dict:
    for where in (directory, os.path.dirname(os.path.abspath(directory))):
        path = os.path.join(where, "manifest.json")
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    return json.load(fh)
            except (OSError, ValueError):
                pass
    return {}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory",
                        help="a recording folder (contains manifest.json), or one segment")
    parser.add_argument("--tolerance-ms", type=float, default=1.0,
                        help="acceptable cross-camera spread (default 1.0)")
    parser.add_argument("--segment", type=int, help="check only this segment number")
    parser.add_argument("--probe", action="store_true",
                        help="also count the frames in the video files with ffprobe")
    parser.add_argument("--verbose", action="store_true",
                        help="list every incomplete frame group and every segment")
    args = parser.parse_args(argv)

    directory = args.directory
    if not os.path.isdir(directory):
        print(f"not a directory: {directory}", file=sys.stderr)
        return 2

    manifest = find_manifest(directory)
    if manifest:
        print(f"clip      : {manifest.get('clip')}")
        print(f"recorded  : {manifest.get('started_at')} -> "
              f"{manifest.get('stopped_at') or '(still running, or not stopped cleanly)'}")
        print(f"sync mode : {manifest.get('sync_mode')} ({manifest.get('sync_detail')})")
    else:
        print(f"(no manifest.json for {directory})")

    # Exposure timestamps are comparable across cameras only on a shared clock:
    # PTP (ptp_action), or simulated cameras. Otherwise each camera counts from its
    # own power-up, and only arrival times can be compared.
    infos = manifest.get("cameras_info", {})
    simulated = bool(infos) and all(i.get("device_class") == "Dummy"
                                    for i in infos.values())
    shared_clock = manifest.get("sync_mode") == "ptp_action" or simulated or not manifest
    fps = float((manifest.get("config", {}).get("acquisition", {}) or {}).get("fps") or 0)

    segs = segment_dirs(directory)
    if len(segs) > 1 or segs[0] != directory:
        print(f"segments  : {len(segs)}" + (f" (checking segment {args.segment})"
                                            if args.segment is not None else ""))

    def progress(name: str) -> None:
        if len(segs) > 3:
            sys.stderr.write(f"\r  read {name} ")
            sys.stderr.flush()

    rep = check_recording(directory, args.tolerance_ms, shared_clock, args.segment,
                          progress)
    if len(segs) > 3:
        sys.stderr.write("\r" + " " * 40 + "\r")
    if not rep["serials"]:
        print("no per-frame CSVs found -- was encoder.write_metadata_csv disabled?",
              file=sys.stderr)
        return 2

    first = rep["first_key"] or 0
    triggers = rep["last_key"] - first + 1 if rep["last_key"] >= 0 else 0
    print(f"\ncameras   : {len(rep['serials'])}")
    print(f"{'camera':<20}{'real':>10}{'filled':>9}{'missed':>9}{'dropped':>9}"
          f"{'real fps':>10}{'interval ms':>13}{'jitter ms':>11}")
    for s in rep["serials"]:
        mean, worst = rep["intervals"].get(s, (0.0, 0.0))
        real = rep["counts"].get(s, 0)
        real_fps = f"{fps * real / triggers:.2f}" if fps and triggers else "-"
        print(f"{s:<20}{real:>10}{rep['placeholders'].get(s, 0):>9}"
              f"{rep['camera_gaps'].get(s, 0):>9}{rep['not_written'].get(s, 0):>9}"
              f"{real_fps:>10}{mean:>13.3f}{worst:>11.3f}")
    print("  real = frames with an image; filled = placeholders in the video;")
    print("  missed = lost between camera and PC; dropped = discarded on the PC")

    if len(rep["segments"]) > 1:
        shown = rep["segments"] if args.verbose else \
            [g for g in rep["segments"] if g["over"] or g["placeholders"]][:20]
        if shown:
            print(f"\n{'segment':<16}{'groups':>9}{'complete':>10}{'filled':>9}"
                  f"{'max ms':>10}{'over':>7}")
            for g in shown:
                print(f"{g['name']:<16}{g['groups']:>9}{g['complete']:>10}"
                      f"{g['placeholders']:>9}{g['max_ms']:>10.4f}{g['over']:>7}")
            if not args.verbose:
                print("  (segments with filled frames or sync problems; --verbose for all)")

    spread = rep["spread_ms"]
    device = rep["clock_source"] == "device"
    print(f"\ncomplete frame groups : {rep['complete_groups']} / {rep['total_groups']}")
    print(f"cross-camera spread   : mean {spread['mean']:.4f} ms | "
          f"median {spread['median']:.4f} | p95 {spread['p95']:.4f} | "
          f"max {spread['max']:.4f}")
    print(f"groups over {args.tolerance_ms} ms      : {rep['over_tolerance']}")
    print(f"measured from         : {rep['clock_source']} clocks")
    if not device:
        print("  NOTE: the cameras' own timestamps are not on a shared clock in this")
        print("        sync mode (no PTP), so the spread is measured from host arrival")
        print("        times. These include transfer delay and overstate the true")
        print("        exposure spread; the tolerance check is not applied to them.")

    if rep["incomplete"]:
        print(f"\n{rep['incomplete_count']} frame group(s) are missing a camera:")
        items = sorted(rep["incomplete"].items())
        for key, missing in (items if args.verbose else items[:10]):
            print(f"  trigger {key}: missing {', '.join(missing)}")
        if not args.verbose and rep["incomplete_count"] > 10:
            print(f"  ... {rep['incomplete_count'] - 10} more (use --verbose)")

    if args.probe:
        print("\nffprobe (frames in the video files, placeholders included):")
        csv_frames = {s: rep["counts"].get(s, 0) + rep["placeholders"].get(s, 0)
                      for s in rep["serials"]}
        for serial, info in sorted(probe_videos(directory, args.segment).items()):
            note = "" if info["frames"] == csv_frames.get(serial) else \
                f"   ! CSV says {csv_frames.get(serial)}"
            print(f"  {serial:<20} {info['frames']:>9} frames in {info['files']} file(s)  "
                  f"{info.get('size', '?'):>10}  {info.get('codec', '?'):<6} "
                  f"{info['bytes'] / 1e9:8.2f} GB{note}")
            for err in info.get("errors", []):
                print(f"      ERROR {err}")

    sync_ok = not (device and rep["over_tolerance"])
    total_slots = triggers * len(rep["serials"])
    missing = sum(rep["placeholders"].values())
    print("\nVERDICT")
    if device:
        print("  SYNC         " + (f"OK - every complete frame group within "
                                   f"{args.tolerance_ms} ms" if sync_ok else
                                   f"PROBLEM - {rep['over_tolerance']} frame groups over "
                                   f"{args.tolerance_ms} ms"))
    else:
        print("  SYNC         not measurable without a shared camera clock (see NOTE)")
    if total_slots:
        share = missing / total_slots
        print(f"  COMPLETENESS {'OK' if missing == 0 else 'frames missing'} - "
              f"{missing} of {total_slots} camera-frames are placeholders ({share:.2%})")
    return 0 if sync_ok else 1


if __name__ == "__main__":
    sys.exit(main())
