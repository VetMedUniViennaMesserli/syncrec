"""Find out why Basler cameras are detected but deliver no images.

Tests each link in the chain separately, cheapest first, and stops at the first one
that breaks:

  1. free run         -- does image data reach this PC at all? (network, driver,
                         firewall, packet size / jumbo frames)
  2. software trigger -- can the host trigger the camera over its control channel?
  3. action command   -- do broadcast action commands reach the cameras, and via
                         which broadcast address? (what ptp_action depends on)
  4. scheduled action -- (--ptp) PTP lock + an action scheduled in the future

    python tools/diagnose_pylon.py
    python tools/diagnose_pylon.py --ptp
    python tools/diagnose_pylon.py -c my_config.json --serial <serial>

Close pylon Viewer and the recorder first: a camera can only be opened once.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pypylon import pylon                                    # noqa: E402

from syncrec.backends.pylon_backend import (PylonCamera, discover_pylon,  # noqa: E402
                                           gige_transport_layer)
from syncrec.config import RecorderConfig                     # noqa: E402

OK, FAIL = "  OK  ", " FAIL "


def grab_one(cam: PylonCamera, timeout_ms: int = 3000, fire=None) -> str:
    """Start grabbing, optionally fire a trigger, and report what came back."""
    cam.start()
    try:
        time.sleep(0.05)
        if fire is not None:
            fire()
        t0 = time.time()
        while time.time() - t0 < timeout_ms / 1000:
            res = cam._cam.RetrieveResult(200, pylon.TimeoutHandling_Return)
            if res is None or not res.IsValid():
                continue
            try:
                if res.GrabSucceeded():
                    return "image"
                return f"failed: {res.GetErrorDescription()}"
            finally:
                res.Release()
        return "nothing"
    finally:
        cam.stop()


def test_freerun(cams, packet_size: int) -> bool:
    print(f"\n[1] Free run, packet size {packet_size}")
    all_ok = True
    for cam in cams:
        cam.set_trigger("off")
        cam.set_param("GevSCPSPacketSize", packet_size)
        got = grab_one(cam)
        ok = got == "image"
        all_ok &= ok
        print(f"  [{OK if ok else FAIL}] {cam.info.serial} ({cam.info.ip_address}): {got}")
    return all_ok


def test_software(cams) -> bool:
    print("\n[2] Software trigger")
    all_ok = True
    for cam in cams:
        cam.set_trigger("on", source="Software")
        got = grab_one(cam, fire=lambda c=cam: (c.wait_for_trigger_ready(500),
                                                c.execute_software_trigger()))
        cam.set_trigger("off")
        ok = got == "image"
        all_ok &= ok
        print(f"  [{OK if ok else FAIL}] {cam.info.serial}: {got}")
    return all_ok


def arm_action(cams, cfg) -> None:
    for cam in cams:
        cam.configure_action_keys(cfg.sync.ptp.device_key, cfg.sync.ptp.group_key,
                                  cfg.sync.ptp.group_mask)
        cam.set_trigger("on", source="Action1")


def test_action(cams, cfg, tl) -> list:
    print("\n[3] Action command (unscheduled), per broadcast address")
    arm_action(cams, cfg)
    p = cfg.sync.ptp
    candidates = []
    for cam in cams:
        addr = cam.subnet_broadcast()
        if addr and addr not in candidates:
            candidates.append(addr)
    candidates.append("255.255.255.255")
    working = []
    for addr in candidates:
        for cam in cams:
            cam.start()
        time.sleep(0.2)
        try:
            _ok, results = tl.IssueActionCommandWait(
                p.device_key, p.group_key, p.group_mask, addr, 500, len(cams))
        except Exception as exc:
            print(f"  [{FAIL}] {addr}: {exc}")
            results = ()
        acks = {ip: int(st) for ip, st in (results or ()) if ip and ip != "0.0.0.0"}
        images, per_cam = 0, {}
        for cam in cams:
            res = cam._cam.RetrieveResult(1000, pylon.TimeoutHandling_Return)
            got = "nothing"
            if res is not None and res.IsValid():
                ok_grab = res.GrabSucceeded()
                images += int(ok_grab)
                got = "image" if ok_grab else f"failed: {res.GetErrorDescription()}"
                res.Release()
            stats = cam.statistics()
            if stats and got != "image":
                got += "  [" + ", ".join(f"{k}={v}" for k, v in stats.items()) + "]"
            per_cam[cam.info.serial] = got
            cam.stop()
        ok = images == len(cams)
        if ok:
            working.append(addr)
        print(f"  [{OK if ok else FAIL}] {addr:<16} acknowledged by {len(acks)}/"
              f"{len(cams)}, images from {images}/{len(cams)}")
        if not ok:
            for cam in cams:
                st = acks.get(cam.info.ip_address)
                st_txt = "no ack" if st is None else f"ack status 0x{st:04X}"
                print(f"         {cam.info.serial}: {st_txt}, {per_cam[cam.info.serial]}")
    if not working:
        dump_trigger_state(cams)
    return working


def dump_trigger_state(cams) -> None:
    """Read back everything that decides whether an action command fires a frame."""
    print("\n    Trigger/action state read back from the cameras:")
    for cam in cams:
        g = cam.get_param
        sel_node = cam._first_node("ActionSelector")
        sel_range = ""
        if sel_node is not None:
            try:
                sel_range = f" (range {sel_node.GetMin()}..{sel_node.GetMax()})"
            except Exception:
                pass
        trig = []
        for sel in cam._enum_entries("TriggerSelector"):
            cam.set_param("TriggerSelector", sel)
            trig.append(f"{sel}={g('TriggerMode')}/{g('TriggerSource')}")
        cam.set_param("TriggerSelector", "FrameStart")
        def hx(v):
            return f"0x{int(v):X}" if isinstance(v, int) else str(v)
        print(f"    {cam.info.serial}: ActionSelector={g('ActionSelector')}{sel_range} "
              f"GroupKey={hx(g('ActionGroupKey'))} GroupMask={hx(g('ActionGroupMask'))} "
              f"PTP={g('GevIEEE1588')} AcqMode={g('AcquisitionMode')}")
        print(f"      triggers: {', '.join(trig)}")
        print(f"      sources:  {', '.join(cam._enum_entries('TriggerSource'))}")


def test_scheduled(cams, cfg, tl, addr: str) -> None:
    print(f"\n[4] PTP + scheduled action command via {addr}")
    for cam in cams:
        cam.set_trigger("off")
        cam.configure_ptp(True)
    deadline = time.time() + cfg.sync.ptp.sync_timeout_s
    t0 = time.time()
    while True:
        states = {c.info.serial: c.ptp_status() for c in cams}
        roles = [str(s["state"]) for s in states.values()]
        slave_offs = [abs(s["offset_ns"]) for s in states.values()
                      if s["state"] == "Slave" and s["offset_ns"] is not None]
        if (roles.count("Master") == 1 and roles.count("Slave") == len(cams) - 1
                and (not slave_offs or max(slave_offs) <= cfg.sync.ptp.max_offset_ns)):
            print(f"         locked after {time.time() - t0:.0f}s")
            break
        if time.time() > deadline:
            print(f"  [{FAIL}] PTP did not lock: {states}")
            return
        time.sleep(0.5)
    for serial, st in states.items():
        print(f"         {serial}: {st['state']:<8} offset {st['offset_ns']} ns")
    freq = cams[0].get_param("GevTimestampTickFrequency")
    print(f"         timestamp tick frequency: {freq}")
    arm_action(cams, cfg)
    for cam in cams:
        cam.start()
    now = cams[0].device_timestamp_ns()
    p = cfg.sync.ptp
    _ok, results = tl.IssueScheduledActionCommandWait(
        p.device_key, p.group_key, p.group_mask, int(now + 200_000_000), addr, 500,
        len(cams))
    time.sleep(0.4)
    images = 0
    for cam in cams:
        res = cam._cam.RetrieveResult(1000, pylon.TimeoutHandling_Return)
        if res is not None and res.IsValid():
            images += int(res.GrabSucceeded())
            res.Release()
        cam.stop()
        cam.set_trigger("off")
    acked = {ip for ip, _st in (results or ()) if ip and ip != "0.0.0.0"}
    ok = images == len(cams)
    print(f"  [{OK if ok else FAIL}] acknowledged by {len(acked)}/{len(cams)}, "
          f"images from {images}/{len(cams)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", help="config JSON (default: built-in defaults)")
    ap.add_argument("--serial", action="append", help="test only this camera")
    ap.add_argument("--ptp", action="store_true", help="also test PTP + scheduled action")
    args = ap.parse_args()

    cfg = RecorderConfig.load(args.config) if args.config else RecorderConfig()
    cams = [c for c in discover_pylon(cfg) if not args.serial or c.info.serial in args.serial]
    if not cams:
        print("No Basler cameras found.")
        return 1
    print(f"{len(cams)} camera(s):")
    for cam in cams:
        cam.open()
        # Same PTP state on every camera. With --ptp leave it on, so a lock the
        # cameras already have is kept (and the recorder can reuse it afterwards).
        cam.configure_ptp(bool(args.ptp))
        cam.configure(cfg)
        w, h, fmt = cam.output_geometry()
        print(f"  {cam.info.serial}  {cam.info.model}  {cam.info.ip_address}  "
              f"{w}x{h} {cam.get_param('PixelFormat')}  "
              f"packet {cam.get_param('GevSCPSPacketSize')}  "
              f"delay {cam.get_param('GevSCPD')} ticks @ "
              f"{(cam.get_param('GevTimestampTickFrequency') or 0) / 1e6:.0f} MHz  "
              f"PTP {cam.get_param('GevIEEE1588')}  "
              f"action queue {cam.action_queue_size() or '? (assumed 1)'}")
    try:
        if not test_freerun(cams, cfg.acquisition.packet_size):
            if cfg.acquisition.packet_size > 1500 and test_freerun(cams, 1500):
                print("\n=> Images arrive with 1500-byte packets but not with "
                      f"{cfg.acquisition.packet_size}: jumbo frames are not enabled on "
                      "the NIC (or switch). Enable them (9014 bytes) or set "
                      "acquisition.packet_size to 1500.")
            else:
                print("\n=> No image data reaches this PC even in free run. This is not "
                      "a sync problem. Check: pylon GigE Performance/Filter driver "
                      "bound to the camera NIC (pylon GigE Configurator), Windows "
                      "firewall allowing python.exe, and that pylon Viewer shows live "
                      "images for these cameras.")
            return 2
        if not test_software(cams):
            print("\n=> Free run works but software triggering does not.")
            return 3
        if not all(c.info.is_gige for c in cams):
            print("\nNot all cameras are GigE: action commands / PTP do not apply. "
                  "Use hardware_trigger or software_trigger.")
            return 0
        tl = gige_transport_layer()
        if tl is None:
            print("\nNo GigE transport layer -- cannot test action commands.")
            return 4
        working = test_action(cams, cfg, tl)
        if not working:
            print("\n=> No images from action commands. If the cameras acknowledged, "
                  "the command arrives but does not fire the trigger -- see the state "
                  "dump above. If not, it never reaches them.")
            return 5
        if "255.255.255.255" not in working:
            print(f"\n=> 255.255.255.255 does not reach the cameras, {working[0]} does. "
                  f"Use sync.ptp.broadcast_address \"auto\" (or \"{working[0]}\").")
        if args.ptp:
            test_scheduled(cams, cfg, tl, working[0])
        print("\nDone.")
        return 0
    finally:
        for cam in cams:
            try:
                cam.set_trigger("off")
                cam.close()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
