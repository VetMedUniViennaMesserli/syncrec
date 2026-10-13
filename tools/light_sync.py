"""Check that a recording is synchronised, using a light switched on and off.

Every camera sees the room go dark at the same physical moment, so in synchronised
videos the first dark frame has the same frame number in every file. This tool
measures the mean brightness of every frame, finds the brightness drops (light off)
and rises (light on), and compares the frame numbers across cameras.

    python tools/light_sync.py recordings/20261012_141500
    python tools/light_sync.py recordings/20261012_141500 --plot
    python tools/light_sync.py recordings/20261012_141500 --segment 2

Use the light-OFF events for the verdict. Switching on is often uneven (several
lamps, lamps warming up); switching off is the sharp edge. On events are reported
too, for information.

The per-frame CSV the recorder writes next to each video is used to flag events
that fall on placeholder frames (a missed frame filled with the previous image):
there the event frame can appear late, so those events are not counted as proof
either way.

Resolution: one frame is the unit here. If the switch happens while the cameras are
exposing, the frame is partly dark; the "partial" column then shows how dark (0 =
still fully lit, 1 = fully dark). Synchronised global-shutter cameras show nearly
the same value; a difference d means the exposures were offset by about
d x exposure time.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

ANALYSIS_WIDTH = 64      # frames are downscaled for speed; mean brightness survives


def brightness(path: str, ffmpeg: str) -> np.ndarray:
    """Mean grey level (0-255) of every frame of a video."""
    probe = subprocess.run(
        [ffmpeg, "-v", "error", "-i", path, "-vf",
         f"scale={ANALYSIS_WIDTH}:-2,format=gray", "-f", "rawvideo", "-"],
        capture_output=True)
    if probe.returncode != 0:
        raise RuntimeError(f"ffmpeg could not read {path}: "
                           f"{probe.stderr.decode(errors='replace')[-300:]}")
    data = np.frombuffer(probe.stdout, dtype=np.uint8)
    # Recover the frame height from the byte count of one frame.
    info = subprocess.run(
        [ffmpeg, "-v", "error", "-i", path, "-vf",
         f"scale={ANALYSIS_WIDTH}:-2,format=gray", "-frames:v", "1",
         "-f", "rawvideo", "-"], capture_output=True).stdout
    frame_bytes = len(info)
    if frame_bytes == 0 or len(data) % frame_bytes:
        raise RuntimeError(f"unexpected decoded size for {path}")
    return data.reshape(-1, frame_bytes).mean(axis=1)


def read_placeholders(csv_path: str) -> Optional[np.ndarray]:
    """Boolean per video frame: True where the recorder filled in a missed frame."""
    if not os.path.exists(csv_path):
        return None
    flags: List[bool] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("written") == "0":
                continue                     # dropped, never reached the video
            flags.append(row.get("placeholder") == "1")
    return np.array(flags, dtype=bool)


def find_events(b: np.ndarray, min_step: float) -> List[Tuple[str, int, float]]:
    """Light edges as (kind, first frame at the new level, partial fraction).

    A drop is "off", a rise is "on". The event frame is the first one past the
    midpoint between the levels before and after the edge. If one frame sits in
    between (light switched during its exposure), partial says how far it got.

    The levels are medians of a few frames on either side, but never reaching past
    a neighbouring edge (light switched back quickly) -- otherwise they mix two
    levels. Lamps that keep fading after the switch can leave no frame past the
    midpoint; then the frame after the biggest single step is taken.
    """
    events: List[Tuple[str, int, float]] = []
    d = np.diff(b)
    big = np.abs(d) >= min_step / 3
    prev_end = 0
    i = 0
    while i < len(d):
        if not big[i]:
            i += 1
            continue
        # Collect a run of steps in the same direction (an edge spanning frames).
        sign = np.sign(d[i])
        j = i
        while j + 1 < len(d) and np.sign(d[j + 1]) == sign and abs(d[j + 1]) >= min_step / 10:
            j += 1
        # Level windows: up to 5 frames, stopping at the previous / next edge.
        lo = max(prev_end, i - 5)
        hi = j + 6
        nxt = np.nonzero(big[j + 1:hi])[0]
        if nxt.size:
            hi = j + 1 + int(nxt[0]) + 1
        before = float(np.median(b[lo:i + 1]))
        after = float(np.median(b[j + 1:hi])) if hi > j + 1 else float(b[j + 1])
        if abs(after - before) >= min_step:
            mid = (before + after) / 2
            seg = b[i:hi]
            crossed = np.nonzero(seg < mid)[0] if sign < 0 else np.nonzero(seg > mid)[0]
            if crossed.size:
                first = i + int(crossed[0])
            else:
                first = i + 1 + int(np.argmax(np.abs(d[i:j + 1])))
            # Partial: how far the frame just before the new level had moved.
            partial = float("nan")
            if first - 1 > i:
                k = first - 1
                partial = float(np.clip((b[k] - before) / (after - before), 0, 1))
            events.append(("off" if sign < 0 else "on", first, partial))
        prev_end = j + 1
        i = j + 1
    return events


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", help="a recording folder (with the .mp4 and .csv files)")
    ap.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    ap.add_argument("--min-step", type=float, default=0.3,
                    help="smallest edge to count, as a fraction of the brightness "
                         "range of the video (default 0.3)")
    ap.add_argument("--plot", action="store_true",
                    help="save light_sync.png with the brightness curves")
    ap.add_argument("--segment", type=int,
                    help="split recording: analyse only this segment (default: all, "
                         "frame numbers then count from the start of the recording)")
    args = ap.parse_args()

    # Split recordings keep each segment in its own folder; the segments of one
    # camera are joined here, so frame numbers are those of the whole recording.
    folders = sorted(d for d in glob.glob(os.path.join(args.directory, "segment_*"))
                     if os.path.isdir(d)) or [args.directory]
    if args.segment is not None:
        folders = [f for f in folders
                   if os.path.basename(f) == f"segment_{args.segment:04d}"]
    per_cam: Dict[str, List[str]] = {}
    for folder in folders:
        for path in sorted(p for p in glob.glob(os.path.join(folder, "*"))
                           if p.lower().endswith((".mp4", ".mkv", ".avi", ".mov"))):
            name = os.path.splitext(os.path.basename(path))[0].rsplit("_", 1)[-1]
            per_cam.setdefault(name, []).append(path)
    if not per_cam:
        print(f"No videos in {args.directory}")
        return 1
    if len(folders) > 1:
        print(f"{len(folders)} segments")

    curves: Dict[str, np.ndarray] = {}
    events: Dict[str, List[Tuple[str, int, float]]] = {}
    fillers: Dict[str, Optional[np.ndarray]] = {}
    for name, paths in per_cam.items():
        parts, flags = [], []
        for path in paths:
            b_part = brightness(path, args.ffmpeg)
            parts.append(b_part)
            fl = read_placeholders(os.path.splitext(path)[0] + ".csv")
            flags.append(fl[:len(b_part)] if fl is not None and len(fl) >= len(b_part)
                         else np.zeros(len(b_part), bool) if fl is None
                         else np.concatenate([fl, np.zeros(len(b_part) - len(fl), bool)]))
        b = np.concatenate(parts)
        rng = float(np.percentile(b, 99) - np.percentile(b, 1))
        curves[name] = b
        events[name] = find_events(b, max(rng * args.min_step, 2.0)) if rng > 0 else []
        fillers[name] = np.concatenate(flags)
        n_on = sum(1 for e in events[name] if e[0] == "on")
        n_off = sum(1 for e in events[name] if e[0] == "off")
        print(f"{name}: {len(b)} frames, brightness {b.min():.0f}-{b.max():.0f}, "
              f"{n_on} on / {n_off} off")

    names = list(curves)
    ok = True
    window = 12          # frames: events further apart than this are different events
    for kind in ("off", "on"):
        per_cam = {n: [e for e in events[n] if e[0] == kind] for n in names}
        if not any(per_cam.values()):
            continue
        # Match events across cameras by position, not by order: one camera seeing
        # an extra edge (someone walking past) must not shift all later pairs.
        clusters: List[Dict[str, Tuple[int, float]]] = []
        for n in names:
            for _, frame, partial in per_cam[n]:
                home = next((c for c in clusters if n not in c and
                             abs(np.median([f for f, _ in c.values()]) - frame) <= window),
                            None)
                if home is None:
                    home = {}
                    clusters.append(home)
                home[n] = (frame, partial)
        clusters.sort(key=lambda c: min(f for f, _ in c.values()))
        for k, c in enumerate(clusters):
            print(f"\nLight-{kind} #{k + 1}:")
            if len(c) < len(names) / 2:
                print(f"  seen by {', '.join(c)} only - a local change (e.g. someone in "
                      f"front of the camera), not the light; ignored")
                continue
            solid = []
            for n in names:
                if n not in c:
                    print(f"  {n}: no edge found here")
                    continue
                frame, partial = c[n]
                fl = fillers[n]
                near_filler = fl is not None and bool(fl[max(0, frame - 2):frame + 1].any())
                p = "" if np.isnan(partial) else f"  partial {partial:.2f}"
                flag = "  (placeholder frame nearby - not conclusive)" if near_filler else ""
                print(f"  {n}: first frame at new level = {frame}{p}{flag}")
                if not near_filler:
                    solid.append(frame)
            if not solid:
                print("  -> no conclusive camera for this event")
                continue
            spread = max(solid) - min(solid)
            if spread == 0:
                print(f"  -> all conclusive cameras agree (frame {solid[0]})")
            else:
                print(f"  -> SPREAD OF {spread} FRAME(S)")
                if kind == "off":
                    ok = False

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("\n--plot needs matplotlib (pip install matplotlib)")
        else:
            fig, ax = plt.subplots(figsize=(12, 5))
            for n in names:
                b = curves[n]
                lo, hi = np.percentile(b, 1), np.percentile(b, 99)
                ax.plot((b - lo) / max(hi - lo, 1e-6), label=n, lw=1)
                fl = fillers[n]
                if fl is not None and fl.any():
                    idx = np.where(fl[:len(b)])[0]
                    ax.plot(idx, ((b - lo) / max(hi - lo, 1e-6))[idx], "x", ms=4)
            ax.set_xlabel("video frame")
            ax.set_ylabel("brightness (normalised per camera)")
            ax.set_title("Light switching - edges should line up (x = placeholder)")
            ax.legend(fontsize=7, ncol=4)
            out = os.path.join(args.directory, "light_sync.png")
            fig.tight_layout()
            fig.savefig(out, dpi=120)
            print(f"\nPlot saved to {out} (zoom in around the edges)")

    if not any(e[0] == "off" for evs in events.values() for e in evs):
        print("\nRESULT: no light-off edge found - nothing to compare (switch the light "
              "off during the recording, or lower --min-step)")
        return 3
    print("\nRESULT:", "light-off edges line up in every camera" if ok
          else "cameras DISAGREE on a light-off edge")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
