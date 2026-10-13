"""Inspect this PC and the connected Basler cameras, and recommend recording settings.

Works on Windows and Linux. Reads, without changing anything permanent:

  - the PC: operating system, CPU, memory, GPUs
  - every network adapter: link speed, MTU / jumbo frames, driver, USB or PCIe,
    addresses, receive buffers, firewall and network profile (Windows),
    socket buffer limit (Linux)
  - ffmpeg: version, and which codecs / GPU encoders really work here, tested with
    one encode per camera running at once
  - the recording disk: free space and write speed
  - every camera: model, serial, firmware, IP, sensor size, pixel formats, link
    speed, PTP / action command support, and the frame rate the camera itself can
    reach at several image sizes (measured by setting each size briefly; all
    settings are restored afterwards)

Then it works out what the hardware can record and prints recommended settings:
image size, inter-packet delay, packet size, sync mode, encoder, and the OS
changes worth making. With --write-config the recommendation is saved as a
config file for the recorder.

    python -m syncrec check                             (25 fps target)
    python -m syncrec check --fps 30 -c config.json --write-config recommended.json
    python -m syncrec check --mode full_res --write-config fullres.json
    python -m syncrec check --mode fewer_cameras --write-config fewer.json
    python -m syncrec check --quick                     (skip encoder, disk, CPU tests)
    python -m syncrec check --json report.json          (everything, machine-readable)

(or the same options with: python tools/system_check.py)

Close the recorder and pylon Viewer first: a camera can only be opened once.
Some details need administrator rights to read; the script says what it could not see.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import math
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from syncrec.config import RecorderConfig                      # noqa: E402

IS_WINDOWS = sys.platform == "win32"
IS_LINUX = sys.platform.startswith("linux")

# Share of a link the recommendation plans to use. The rest is headroom for
# resends, PTP and control traffic, and the bursts of synchronised cameras.
LINK_BUDGET = 0.65
CAMERA_LINK_BUDGET = 0.85        # of the camera's own 1 Gb/s link
BURST_BUDGET = 0.70              # of the host link while all cameras send at once
TRANSFER_BUDGET = 0.80           # of the frame period one frame may take to send
# Prefer the whole field of view (full sensor or binned) over a larger cut-out.
# Set to False by --prefer-pixels.
PREFER_FULL_VIEW = True
GVSP_HEADERS = 36                # bytes of IP/UDP/GVSP header per packet
WIRE_OVERHEAD = 38               # Ethernet preamble, header, FCS, gap


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def run(cmd, timeout: float = 20.0, shell: bool = False) -> Tuple[int, str]:
    """Run a command; (returncode, stdout). Never raises."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           shell=shell, errors="replace",
                           creationflags=subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0)
        return r.returncode, r.stdout
    except (OSError, subprocess.SubprocessError) as exc:
        return -1, str(exc)


def powershell_json(script: str, timeout: float = 30.0) -> Any:
    """Run PowerShell, return its ConvertTo-Json output parsed (lists always lists)."""
    rc, out = run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                   f"$ErrorActionPreference='SilentlyContinue'; {script} | "
                   f"ConvertTo-Json -Depth 4 -Compress"], timeout=timeout)
    if rc != 0 or not out.strip():
        return []
    try:
        data = json.loads(out)
    except ValueError:
        return []
    return data if isinstance(data, list) else [data]


def read_file(path: str) -> Optional[str]:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read().strip()
    except OSError:
        return None


def round_down(value: float, step: int) -> int:
    return int(value // step * step)


# ---------------------------------------------------------------------------
# the PC
# ---------------------------------------------------------------------------
def collect_system() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "logical_cpus": os.cpu_count() or 1,
    }
    if IS_LINUX:
        cpuinfo = read_file("/proc/cpuinfo") or ""
        m = re.search(r"model name\s*:\s*(.+)", cpuinfo)
        info["cpu"] = m.group(1).strip() if m else platform.processor()
        cores = {(p, c) for p, c in re.findall(
            r"physical id\s*:\s*(\d+).*?core id\s*:\s*(\d+)", cpuinfo, re.S)}
        info["physical_cores"] = len(cores) or info["logical_cpus"]
        m = re.search(r"MemTotal:\s*(\d+)", read_file("/proc/meminfo") or "")
        info["ram_gb"] = round(int(m.group(1)) / 1e6, 1) if m else None
        rel = read_file("/etc/os-release") or ""
        m = re.search(r'PRETTY_NAME="([^"]+)"', rel)
        if m:
            info["os"] = m.group(1) + f" (kernel {platform.release()})"
    elif IS_WINDOWS:
        cpu = powershell_json("Get-CimInstance Win32_Processor | "
                              "Select-Object Name,NumberOfCores,NumberOfLogicalProcessors")
        if cpu:
            info["cpu"] = cpu[0].get("Name", "").strip()
            info["physical_cores"] = sum(int(c.get("NumberOfCores") or 0) for c in cpu)
        mem = powershell_json("Get-CimInstance Win32_ComputerSystem | "
                              "Select-Object TotalPhysicalMemory")
        if mem:
            info["ram_gb"] = round(int(mem[0].get("TotalPhysicalMemory") or 0) / 1e9, 1)
    info.setdefault("cpu", platform.processor() or "unknown")
    info.setdefault("physical_cores", info["logical_cpus"])
    for mod in ("pypylon", "PySide6", "numpy"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "installed")
        except Exception:
            info[mod] = None
    return info


def collect_gpus() -> List[Dict[str, Any]]:
    gpus: List[Dict[str, Any]] = []
    if shutil.which("nvidia-smi"):
        rc, out = run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                       "--format=csv,noheader"])
        for line in out.splitlines() if rc == 0 else []:
            parts = [p.strip() for p in line.split(",")]
            if parts and parts[0]:
                gpus.append({"vendor": "NVIDIA", "name": parts[0],
                             "driver": parts[1] if len(parts) > 1 else "",
                             "memory": parts[2] if len(parts) > 2 else ""})
    if IS_WINDOWS:
        for g in powershell_json("Get-CimInstance Win32_VideoController | "
                                 "Select-Object Name,DriverVersion"):
            name = (g.get("Name") or "").strip()
            if name and not any(name == x["name"] for x in gpus):
                vendor = ("NVIDIA" if "NVIDIA" in name.upper() else
                          "Intel" if "INTEL" in name.upper() else
                          "AMD" if any(k in name.upper() for k in ("AMD", "RADEON")) else "")
                gpus.append({"vendor": vendor, "name": name,
                             "driver": g.get("DriverVersion", "")})
    elif IS_LINUX:
        if shutil.which("lspci"):
            rc, out = run(["lspci"])
            for line in out.splitlines():
                if re.search(r"VGA|3D controller|Display controller", line):
                    name = line.split(":", 2)[-1].strip()
                    if not any(x["vendor"] == "NVIDIA" and "NVIDIA" in name.upper()
                               for x in gpus):
                        gpus.append({"vendor": "", "name": name, "driver": ""})
        nodes = sorted(n for n in os.listdir("/dev/dri") if n.startswith("renderD")) \
            if os.path.isdir("/dev/dri") else []
        for g in gpus:
            g["render_nodes"] = nodes
        if not gpus and nodes:
            gpus.append({"vendor": "", "name": "GPU (render node present)",
                         "render_nodes": nodes})
    return gpus


# ---------------------------------------------------------------------------
# network
# ---------------------------------------------------------------------------
def _linux_ipv4(ifname: str) -> List[Tuple[str, int]]:
    """IPv4 addresses of an interface: `ip -j`, else ioctl (no iproute2 needed)."""
    if shutil.which("ip"):
        rc, out = run(["ip", "-j", "-4", "addr", "show", "dev", ifname])
        try:
            data = json.loads(out) if rc == 0 else []
            return [(a["local"], int(a["prefixlen"])) for d in data
                    for a in d.get("addr_info", []) if a.get("family") == "inet"]
        except (ValueError, KeyError):
            pass
    try:
        import fcntl
        import struct
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        req = struct.pack("256s", ifname[:15].encode())
        ip = socket.inet_ntoa(fcntl.ioctl(s.fileno(), 0x8915, req)[20:24])   # SIOCGIFADDR
        mask = socket.inet_ntoa(fcntl.ioctl(s.fileno(), 0x891b, req)[20:24])  # NETMASK
        return [(ip, ipaddress.IPv4Network(f"0.0.0.0/{mask}").prefixlen)]
    except Exception:
        return []


def _linux_adapters() -> List[Dict[str, Any]]:
    adapters = []
    base = "/sys/class/net"
    for name in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        path = os.path.join(base, name)
        if name == "lo" or not os.path.exists(os.path.join(path, "device")):
            continue                                   # virtual: bridges, docker, ...
        dev = os.path.realpath(os.path.join(path, "device"))
        driver = os.path.basename(os.path.realpath(os.path.join(path, "device", "driver")))
        speed = read_file(os.path.join(path, "speed"))
        a: Dict[str, Any] = {
            "name": name,
            "description": driver,
            "driver": driver,
            "usb": "/usb" in dev,
            "up": read_file(os.path.join(path, "operstate")) == "up",
            "speed_mbps": int(speed) if speed and speed.lstrip("-").isdigit()
            and int(speed) > 0 else None,
            "mtu": int(read_file(os.path.join(path, "mtu")) or 0) or None,
            "ipv4": _linux_ipv4(name),
            "wireless": os.path.isdir(os.path.join(path, "wireless")),
        }
        if shutil.which("ip"):
            rc, out = run(["ip", "-d", "link", "show", name])
            m = re.search(r"maxmtu (\d+)", out)
            if m:
                a["max_mtu"] = int(m.group(1))
        if shutil.which("ethtool"):
            rc, out = run(["ethtool", "-g", name])
            if rc == 0:
                nums = re.findall(r"^RX:\s+(\d+)", out, re.M)
                if len(nums) >= 2:
                    a["rx_ring_max"], a["rx_ring"] = int(nums[0]), int(nums[1])
            rc, out = run(["ethtool", "-i", name])
            m = re.search(r"^driver:\s*(\S+)", out, re.M)
            if m:
                a["driver"] = m.group(1)
        adapters.append(a)
    return adapters


def _windows_adapters() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    host: Dict[str, Any] = {}
    nics = powershell_json(
        "Get-NetAdapter -Physical | Select-Object Name,InterfaceDescription,Status,"
        "LinkSpeed,ReceiveLinkSpeed,PnPDeviceID,ifIndex,DriverDescription,MediaType")
    ips = powershell_json("Get-NetIPAddress -AddressFamily IPv4 | "
                          "Select-Object ifIndex,IPAddress,PrefixLength")
    mtus = powershell_json("Get-NetIPInterface -AddressFamily IPv4 | "
                           "Select-Object ifIndex,NlMtu")
    adv = powershell_json(
        "Get-NetAdapterAdvancedProperty -RegistryKeyword '*JumboPacket','*ReceiveBuffers' "
        "| Select-Object Name,RegistryKeyword,DisplayValue,ValidDisplayValues,"
        "NumericParameterMaxValue")
    profiles = powershell_json("Get-NetConnectionProfile | "
                               "Select-Object InterfaceIndex,NetworkCategory,Name")
    bindings = powershell_json(
        "Get-NetAdapterBinding | Where-Object { $_.DisplayName -match 'Basler|pylon' } "
        "| Select-Object Name,DisplayName,Enabled")
    host["firewall"] = powershell_json(
        "Get-NetFirewallProfile | Select-Object Name,Enabled,DefaultInboundAction")
    categories = {0: "Public", 1: "Private", 2: "DomainAuthenticated"}
    adapters = []
    for n in nics:
        idx = n.get("ifIndex")
        name = n.get("Name", "")
        speed_bps = n.get("ReceiveLinkSpeed")
        a: Dict[str, Any] = {
            "name": name,
            "description": n.get("InterfaceDescription", ""),
            "driver": n.get("DriverDescription", ""),
            "usb": str(n.get("PnPDeviceID", "")).upper().startswith("USB"),
            "up": str(n.get("Status", "")).lower() == "up",
            "speed_mbps": int(speed_bps) // 1_000_000 if speed_bps else None,
            "mtu": next((int(m["NlMtu"]) for m in mtus if m.get("ifIndex") == idx), None),
            "ipv4": [(i["IPAddress"], int(i["PrefixLength"])) for i in ips
                     if i.get("ifIndex") == idx],
            "wireless": "802.11" in str(n.get("MediaType", "")) or
                        "wi-fi" in name.lower() or "wireless" in str(n.get("InterfaceDescription", "")).lower(),
        }
        for p in adv:
            if p.get("Name") != name:
                continue
            if p.get("RegistryKeyword") == "*JumboPacket":
                a["jumbo"] = p.get("DisplayValue")
                a["jumbo_options"] = p.get("ValidDisplayValues")
            elif p.get("RegistryKeyword") == "*ReceiveBuffers":
                a["receive_buffers"] = p.get("DisplayValue")
                a["receive_buffers_max"] = p.get("NumericParameterMaxValue")
        prof = next((p for p in profiles if p.get("InterfaceIndex") == idx), None)
        if prof:
            cat = prof.get("NetworkCategory")
            a["network_category"] = categories.get(cat, cat)
        a["basler_driver"] = [f"{b.get('DisplayName')} ({'on' if b.get('Enabled') else 'off'})"
                              for b in bindings if b.get("Name") == name]
        adapters.append(a)
    return adapters, host


def collect_network() -> Dict[str, Any]:
    if IS_WINDOWS:
        adapters, host = _windows_adapters()
    else:
        adapters, host = _linux_adapters(), {}
        rmem = read_file("/proc/sys/net/core/rmem_max")
        host["rmem_max"] = int(rmem) if rmem and rmem.isdigit() else None
        if shutil.which("ufw"):
            rc, out = run(["ufw", "status"])
            host["ufw"] = out.strip().splitlines()[0] if rc == 0 and out.strip() else \
                "unknown (needs root)"
    return {"adapters": adapters, "host": host}


def adapter_for(ip: str, adapters: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The host adapter whose subnet contains the camera's IP."""
    try:
        cam = ipaddress.IPv4Address(ip)
    except ValueError:
        return None
    for a in adapters:
        for addr, prefix in a.get("ipv4", []):
            try:
                if cam in ipaddress.IPv4Network(f"{addr}/{prefix}", strict=False):
                    return a
            except ValueError:
                continue
    return None


# ---------------------------------------------------------------------------
# cameras
# ---------------------------------------------------------------------------
def scale_candidates(width: int, height: int, w_step: int, h_step: int) -> List[Tuple[int, int]]:
    """Image sizes to try: the full sensor, then smaller centred cut-outs (same shape)."""
    # The full size exactly as the camera reports it (only snapped to its own step):
    # rounding it to a multiple of 32 would turn e.g. a 2064-pixel-wide sensor into
    # a 2048 "cut-out" and lose the full-resolution options.
    sizes = [(round_down(width, w_step), round_down(height, h_step))]
    for s in (0.875, 0.75, 0.625, 0.5, 0.4375, 0.375, 0.3125, 0.25):
        w = round_down(width * s, max(32, w_step))
        h = round_down(height * s, max(16, h_step))
        if w >= 256 and h >= 192 and (w, h) not in sizes:
            sizes.append((w, h))
    return sizes


def probe_camera(cam, exposure_us: float) -> Dict[str, Any]:
    """Read one camera; measure its own frame-rate limit per image size."""
    info: Dict[str, Any] = {}
    r = cam._read
    for key, node in (("firmware", "DeviceFirmwareVersion"), ("vendor", "DeviceVendorName"),
                      ("sensor_width", "SensorWidth"), ("sensor_height", "SensorHeight"),
                      ("width_max", "WidthMax"), ("height_max", "HeightMax"),
                      ("link_speed_mbps", "GevLinkSpeed"),
                      ("packet_size", "GevSCPSPacketSize"),
                      ("timestamp_tick_hz", "GevTimestampTickFrequency"),
                      ("pixel_format", "PixelFormat")):
        val = r(node, None)
        if val is not None:
            info[key] = val
    info["pixel_formats"] = cam._enum_entries("PixelFormat")
    info["action_queue_size"] = cam.action_queue_size()
    info["capabilities"] = sorted(c.value for c in cam.capabilities())
    try:
        info["ptp"] = cam.ptp_status()
    except Exception:
        pass
    width = int(info.get("width_max") or info.get("sensor_width") or 0)
    height = int(info.get("height_max") or info.get("sensor_height") or 0)
    if not width or not height:
        return info

    def inc(name: str) -> int:
        node = cam._first_node(name)
        try:
            return max(1, int(node.GetInc())) if node is not None else 1
        except Exception:
            return 1

    # Remember what we change, so the camera is left as we found it.
    names = ["OffsetX", "OffsetY", "Width", "Height", "PixelFormat"]
    saved = {n: r(n, None) for n in names}
    saved_exp = r("exposure", None)
    saved_bin = int(r("BinningHorizontal", 1) or 1)
    saved_dec = int(r("DecimationHorizontal", 1) or 1)
    support = cam.binning_support() if hasattr(cam, "binning_support") else \
        {"binning": 1, "decimation": 1}
    info["binning_max"], info["decimation_max"] = support["binning"], support["decimation"]
    saved_fps_en = r("fps_enable", None)
    saved_delay = r("packet_delay", None)
    trig = {}
    for sel in cam._enum_entries("TriggerSelector"):
        if cam._write("TriggerSelector", sel):
            trig[sel] = r("TriggerMode", None)
    limits: List[Dict[str, Any]] = []
    try:
        for sel in trig:
            cam._write("TriggerSelector", sel)
            cam._write("TriggerMode", "Off")
        cam._write("fps_enable", False)          # the limiter would hide the real maximum
        # The camera's reported maximum includes how fast it may send. An
        # inter-packet delay left over from the last recording would show up here
        # as a "camera" limit; the link budget is worked out separately below.
        cam._write("packet_delay", 0)
        for fmt in ("Mono8", "BayerRG8", "BayerBG8", "BayerGB8", "BayerGR8"):
            if fmt in info["pixel_formats"]:
                cam._write("PixelFormat", fmt)
                info["probe_pixel_format"] = fmt
                break
        cam._write("exposure", float(exposure_us))
        cam._apply_binning(1)
        # The full size with no binning and no offsets: what the recorder last set
        # (a centred cut-out, binning) can make WidthMax report less.
        cam._write("OffsetX", 0)
        cam._write("OffsetY", 0)
        for key, node in (("width_max", "Width"), ("height_max", "Height")):
            n = cam._first_node(node)
            try:
                info[key] = int(n.GetMax())
            except Exception:
                pass
        width = int(info.get("width_max") or width)
        height = int(info.get("height_max") or height)
        # Cut-outs of the sensor (crop: less field of view).
        for w, h in scale_candidates(width, height, inc("Width"), inc("Height")):
            cam._write("OffsetX", 0)
            cam._write("OffsetY", 0)
            ok = cam._write("Width", w) and cam._write("Height", h)
            fps = r("resulting_fps", None)
            w, h = int(r("Width", w)), int(r("Height", h))
            limits.append({"width": w, "height": h, "binning": 1,
                           "coverage": round(w * h / (width * height), 3),
                           "camera_max_fps": round(float(fps), 2) if fps and ok else None,
                           "payload": int(r("PayloadSize", w * h) or w * h)})
        # Binned / decimated: the whole field of view with fewer pixels.
        for b in range(2, min(4, max(support["binning"], support["decimation"])) + 1):
            try:
                cam._apply_binning(b)
            except Exception:
                break
            cam._write("OffsetX", 0)
            cam._write("OffsetY", 0)
            w = round_down(width // b, max(32, inc("Width")))
            h = round_down(height // b, max(16, inc("Height")))
            ok = cam._write("Width", w) and cam._write("Height", h)
            fps = r("resulting_fps", None)
            limits.append({"width": w, "height": h, "binning": b, "coverage": 1.0,
                           "decimation": support["binning"] < b,
                           "camera_max_fps": round(float(fps), 2) if fps and ok else None,
                           "payload": w * h})
    finally:
        try:
            cam._apply_binning(max(saved_bin, saved_dec))
        except Exception:
            pass
        cam._write("OffsetX", 0)
        cam._write("OffsetY", 0)
        for n in ("PixelFormat", "Width", "Height", "OffsetX", "OffsetY"):
            if saved.get(n) is not None:
                cam._write(n, saved[n])
        if saved_exp is not None:
            cam._write("exposure", saved_exp)
        if saved_fps_en is not None:
            cam._write("fps_enable", bool(saved_fps_en))
        if saved_delay is not None:
            cam._write("packet_delay", saved_delay)
        for sel, mode in trig.items():
            if mode is not None and cam._write("TriggerSelector", sel):
                cam._write("TriggerMode", mode)
    info["frame_rate_limits"] = limits
    info["inc"] = {"width": inc("Width"), "height": inc("Height"),
                   "offset_x": inc("OffsetX"), "offset_y": inc("OffsetY")}
    return info


def collect_cameras(cfg: RecorderConfig, measure: bool = True) -> Dict[str, Any]:
    try:
        from syncrec.backends.pylon_backend import PYLON_AVAILABLE, discover_pylon
    except Exception as exc:
        return {"error": f"cannot load the pylon backend: {exc}", "cameras": []}
    if not PYLON_AVAILABLE:
        return {"error": "pypylon is not installed", "cameras": []}
    try:
        cams = discover_pylon(cfg)
    except Exception as exc:
        return {"error": str(exc), "cameras": []}
    out = []
    for cam in cams:
        entry: Dict[str, Any] = {
            "serial": cam.info.serial, "model": cam.info.model,
            "device_class": cam.info.device_class, "ip": cam.info.ip_address,
            "user_name": cam.info.user_id,
        }
        try:
            di = cam._di
            if di.IsSubnetMaskAvailable():
                entry["subnet_mask"] = di.GetSubnetMask()
            if di.IsMacAddressAvailable():
                entry["mac"] = di.GetMacAddress()
        except Exception:
            pass
        try:
            cam.open()
        except Exception as exc:
            entry["error"] = (f"could not open ({exc}). Is it open in pylon Viewer or "
                              f"the recorder?")
            out.append(entry)
            continue
        try:
            if measure:
                entry.update(probe_camera(cam, cfg.acquisition.exposure_us))
            else:
                entry["capabilities"] = sorted(c.value for c in cam.capabilities())
        except Exception as exc:
            entry["error"] = f"probe failed: {exc}"
        finally:
            try:
                cam.close()
            except Exception:
                pass
        out.append(entry)
    return {"cameras": out}


# ---------------------------------------------------------------------------
# ffmpeg, encoders, CPU, disk
# ---------------------------------------------------------------------------
def collect_ffmpeg(cfg: RecorderConfig, sessions: int, test_encoders: bool) -> Dict[str, Any]:
    from syncrec.writer import resolve_ffmpeg
    try:
        ffmpeg = resolve_ffmpeg(cfg.encoder.ffmpeg)
    except FileNotFoundError as exc:
        return {"error": str(exc)}
    rc, out = run([ffmpeg, "-hide_banner", "-version"])
    info: Dict[str, Any] = {"path": ffmpeg,
                            "version": out.splitlines()[0] if out else "unknown"}
    if test_encoders:
        from syncrec.encoders import survey
        info["encoders"] = [{"codec": c, "hardware": h, "works": ok, "detail": msg}
                            for c, h, ok, msg in survey(ffmpeg, max(1, sessions))]
    return info


def benchmark_cpu_encoder(ffmpeg: str, width: int, height: int,
                          codec_args: List[str], frames: int = 100) -> Optional[float]:
    """Frames per second ONE encoder thread manages at this size (grey input)."""
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
           "-f", "lavfi", "-i", f"testsrc2=size={width}x{height}:rate=25",
           "-vf", "format=gray,noise=alls=12:allf=t", "-frames:v", str(frames)] \
        + codec_args + ["-threads", "1", "-f", "null", "-"]
    t0 = time.perf_counter()
    rc, _ = run(cmd, timeout=120)
    dt = time.perf_counter() - t0
    return frames / dt if rc == 0 and dt > 0 else None


def collect_disk(path: str, test_speed: bool) -> Dict[str, Any]:
    os.makedirs(path, exist_ok=True)
    usage = shutil.disk_usage(path)
    info: Dict[str, Any] = {"path": os.path.abspath(path),
                            "free_gb": round(usage.free / 1e9, 1),
                            "total_gb": round(usage.total / 1e9, 1)}
    if test_speed:
        size, chunk = 512 * 2**20, 8 * 2**20
        block = os.urandom(chunk)
        fd, tmp = tempfile.mkstemp(dir=path, prefix=".speedtest_")
        try:
            t0 = time.perf_counter()
            with os.fdopen(fd, "wb", buffering=0) as fh:
                for _ in range(size // chunk):
                    fh.write(block)
                fh.flush()
                os.fsync(fh.fileno())
            info["write_mb_s"] = round(size / 1e6 / (time.perf_counter() - t0), 0)
        except OSError as exc:
            info["write_error"] = str(exc)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
    return info


# ---------------------------------------------------------------------------
# recommendation
# ---------------------------------------------------------------------------
# Rough compressed sizes in bits per pixel for grey, moderately busy scenes.
BITS_PER_PIXEL = {"h264": 0.12, "h265": 0.07, "av1": 0.06, "mjpeg": 1.5, "ffv1": 4.0}


def plan_group(cams: List[Dict[str, Any]], adapter: Optional[Dict[str, Any]],
               fps: float) -> Dict[str, Any]:
    """Image size and packet timing for the cameras sharing one host adapter."""
    n = len(cams)
    gige = all("gige" in str(c.get("device_class", "")).lower() for c in cams)
    if not gige:
        return plan_usb_group(cams, fps)
    host_mbps = (adapter or {}).get("speed_mbps") or 1000
    cam_mbps = min((int(c.get("link_speed_mbps") or 1000) for c in cams), default=1000)
    jumbo = bool(adapter) and (adapter.get("mtu") or 0) >= 9000
    if adapter and IS_WINDOWS and "jumbo" in adapter:
        # "Jumbo Packet" is e.g. "9014 Bytes" or "Disabled".
        m = re.search(r"\d+", str(adapter.get("jumbo")))
        jumbo = jumbo or bool(m and int(m.group()) >= 9000)
    packet = 8192 if jumbo else 1500
    host_bps = host_mbps * 1e6 / 8
    cam_bps = cam_mbps * 1e6 / 8
    per_cam = min(host_bps * LINK_BUDGET / n, cam_bps * CAMERA_LINK_BUDGET)
    max_frame_bytes = per_cam / fps

    sizes, table, choice = _size_table(cams, per_cam, fps, max_frame_bytes)
    result: Dict[str, Any] = {
        "cameras": [c["serial"] for c in cams], "adapter": (adapter or {}).get("name"),
        "host_link_mbps": host_mbps, "camera_link_mbps": cam_mbps, "jumbo": jumbo,
        "packet_size": packet, "per_camera_budget_mb_s": round(per_cam / 1e6, 1),
        "size_table": table,
    }
    if choice is None:
        result["problem"] = (f"no tested image size reaches {fps:g} fps for {n} cameras "
                             f"on this link")
        _add_plans(result, sizes, cams, fps, host_bps, cam_bps, cam_mbps, packet)
        return result

    result.update(_placement(choice, sizes, cams))
    result.update(_timing(n, choice["payload"], fps, host_bps, cam_mbps, packet))
    _add_plans(result, sizes, cams, fps, host_bps, cam_bps, cam_mbps, packet)
    return result


def _placement(choice: Dict[str, Any], sizes: List[Dict[str, Any]],
               cams: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"roi": _roi_for(choice, sizes, cams[0].get("inc") or {}),
            "binning": choice.get("binning", 1),
            "coverage": choice.get("coverage", 1.0),
            "payload_bytes": choice["payload"]}


def _timing(n: int, payload: int, fps: float, host_bps: float, cam_mbps: float,
            packet: int) -> Dict[str, Any]:
    """Inter-packet delay for n cameras sharing the host link.

    While all cameras send at once they must stay within the burst budget of the
    host link, and one frame must still fit in the frame period.
    """
    data_per_packet = packet - GVSP_HEADERS
    n_packets = math.ceil(payload / data_per_packet)
    wire_s = (packet + WIRE_OVERHEAD) * 8 / (cam_mbps * 1e6)
    interval_s = (packet + WIRE_OVERHEAD) * n / (host_bps * BURST_BUDGET)
    delay_s = max(0.0, interval_s - wire_s)
    period = 1.0 / fps
    if n_packets * (wire_s + delay_s) > TRANSFER_BUDGET * period:
        delay_s = max(0.0, TRANSFER_BUDGET * period / n_packets - wire_s)
    return {
        "inter_packet_delay_us": int(round(delay_s * 1e6 / 10.0) * 10),
        "frame_transfer_ms": round(n_packets * (wire_s + delay_s) * 1e3, 1),
        "burst_link_share": round((packet + WIRE_OVERHEAD) * n / (wire_s + delay_s)
                                  / host_bps, 2),
        "average_link_share": round(payload * fps * n / host_bps, 2),
    }


def nice_fps(value: float) -> float:
    """A round frame rate at or below `value`."""
    if value >= 10:
        return float(math.floor(value))
    if value >= 1:
        return math.floor(value * 2) / 2
    return max(0.1, math.floor(value * 10) / 10)


def _full_sensor(sizes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The largest unbinned size measured: the full sensor.

    Chosen by size rather than by a coverage threshold, so a camera whose reported
    maximum is a few pixels off the sensor size still gets its full-resolution
    options.
    """
    unbinned = [s for s in sizes if s.get("binning", 1) == 1]
    return max(unbinned, key=lambda s: s["width"] * s["height"]) if unbinned else None


def _add_plans(result: Dict[str, Any], sizes: List[Dict[str, Any]],
               cams: List[Dict[str, Any]], fps: float, host_bps: float,
               cam_bps: float, cam_mbps: float, packet: int) -> None:
    """The three ways to use this link: fit / full resolution / fewer cameras."""
    n = len(cams)
    serials = [c["serial"] for c in cams]
    plans: Dict[str, Any] = {}
    if result.get("roi"):
        plans["fit"] = {k: result[k] for k in (
            "roi", "binning", "coverage", "payload_bytes", "inter_packet_delay_us",
            "frame_transfer_ms", "burst_link_share", "average_link_share")}
        plans["fit"].update(fps=fps, cameras=serials)
    full = _full_sensor(sizes)
    if full:
        payload = full["payload"]
        cam_limit = full.get("camera_max_fps") or float("inf")
        place = _placement(full, sizes, cams)

        # B: every camera, full sensor, as fast as the link and the cameras allow.
        per_cam = min(host_bps * LINK_BUDGET / n, cam_bps * CAMERA_LINK_BUDGET)
        best = min(per_cam / payload, cam_limit) / 1.05
        f = nice_fps(best)
        plan = dict(place, fps=f, cameras=serials, max_fps=round(best, 2))
        plan.update(_timing(n, payload, f, host_bps, cam_mbps, packet))
        plans["full_res"] = plan

        # C: full sensor at the target rate, with as many cameras as fit.
        per_cam_one = cam_bps * CAMERA_LINK_BUDGET
        if per_cam_one / payload < fps * 1.05 or cam_limit < fps * 1.05:
            limit = ("its own 1 Gb/s link" if per_cam_one / payload < fps * 1.05
                     else f"the camera itself, which reports at most {cam_limit:g} fps at "
                          f"full size with the current exposure and pixel format")
            plans["fewer_cameras"] = {
                "problem": f"not even one camera reaches {fps:g} fps at full "
                           f"resolution: limited by {limit}"}
        else:
            k = min(n, int(host_bps * LINK_BUDGET // (payload * fps * 1.05)))
            if k == 0:
                plans["fewer_cameras"] = {
                    "problem": f"the PC's link is too slow for even one camera at full "
                               f"resolution and {fps:g} fps"}
                result["plans"] = plans
                return
            plan = dict(place, fps=fps, cameras=serials[:k], dropped=serials[k:])
            plan.update(_timing(max(1, k), payload, fps, host_bps, cam_mbps, packet))
            plans["fewer_cameras"] = plan
    result["plans"] = plans


def _size_table(cams: List[Dict[str, Any]], per_cam: float, fps: float,
                max_frame_bytes: float):
    """Per tested image size: the frame rate the link and the camera allow."""
    by_size: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    for lim in (lim for c in cams for lim in c.get("frame_rate_limits", [])):
        key = (lim["width"], lim["height"], lim.get("binning", 1))
        cur = by_size.setdefault(key, dict(lim))
        if lim.get("camera_max_fps") is not None:
            cur["camera_max_fps"] = min(cur.get("camera_max_fps") or 1e9,
                                        lim["camera_max_fps"])
    # Coverage relative to the largest unbinned size actually measured.
    full = _full_sensor(list(by_size.values()))
    if full:
        area = full["width"] * full["height"]
        for d in by_size.values():
            if d.get("binning", 1) == 1:
                d["coverage"] = round(d["width"] * d["height"] / area, 3)
    # Most pixels first; for the same amount of data, the full view (binned)
    # before a cut-out.
    sizes = sorted(by_size.values(),
                   key=lambda d: (-d["payload"], -d.get("coverage", 1.0)))
    choice, fallback, table = None, None, []
    for s in sizes:
        bw_fps = per_cam / s["payload"]
        cam_fps = s.get("camera_max_fps")
        max_fps = min(bw_fps, cam_fps) if cam_fps else bw_fps
        table.append({"width": s["width"], "height": s["height"],
                      "binning": s.get("binning", 1),
                      "decimation": s.get("decimation", False),
                      "coverage": s.get("coverage", 1.0),
                      "max_fps_bandwidth": round(bw_fps, 1),
                      "max_fps_camera": cam_fps, "max_fps": round(max_fps, 1)})
        fits = max_fps >= fps * 1.05 and s["payload"] <= max_frame_bytes
        if fits and choice is None:
            if not PREFER_FULL_VIEW or s.get("coverage", 1.0) >= 0.999:
                choice = s
        if fits and fallback is None:
            fallback = s
    # No whole-view option fits: the largest cut-out that does.
    return sizes, table, choice or fallback


def _roi_for(choice: Dict[str, Any], sizes: List[Dict[str, Any]],
             incs: Dict[str, int]) -> Dict[str, int]:
    """ROI for the chosen size: centred cut-out, or the whole (binned) image."""
    w, h = choice["width"], choice["height"]
    if choice.get("binning", 1) > 1:
        return {"width": w, "height": h, "offset_x": 0, "offset_y": 0}
    full = [s for s in sizes if s.get("binning", 1) == 1]
    full_w = max(s["width"] for s in full)
    full_h = max(s["height"] for s in full)
    return {"width": w, "height": h,
            "offset_x": round_down((full_w - w) / 2, max(16, incs.get("offset_x", 1))),
            "offset_y": round_down((full_h - h) / 2, max(16, incs.get("offset_y", 1)))}


def plan_usb_group(cams: List[Dict[str, Any]], fps: float) -> Dict[str, Any]:
    """USB3 cameras: no packets or delays; each has its own USB link (~350 MB/s),
    but cameras on one USB controller share it -- not visible from here."""
    per_cam = 350e6 * LINK_BUDGET
    sizes, table, choice = _size_table(cams, per_cam, fps, per_cam / fps)
    result: Dict[str, Any] = {
        "cameras": [c["serial"] for c in cams], "adapter": "USB", "usb": True,
        "host_link_mbps": None, "camera_link_mbps": None, "jumbo": False,
        "packet_size": None, "per_camera_budget_mb_s": round(per_cam / 1e6, 1),
        "size_table": table,
        "note": "USB3: spread the cameras over several USB controllers / cards",
    }
    if choice is None:
        result["problem"] = f"no tested image size reaches {fps:g} fps"
        result["plans"] = {}
        return result
    result.update(_placement(choice, sizes, cams))
    plans: Dict[str, Any] = {"fit": {k: result[k] for k in (
        "roi", "binning", "coverage", "payload_bytes")}}
    plans["fit"].update(fps=fps, cameras=result["cameras"])
    full = _full_sensor(sizes)
    if full:
        best = min(per_cam / full["payload"],
                   full.get("camera_max_fps") or float("inf")) / 1.05
        plans["full_res"] = dict(_placement(full, sizes, cams), fps=nice_fps(best),
                                 cameras=result["cameras"], max_fps=round(best, 2))
        plans["fewer_cameras"] = {"problem": "with USB every camera has its own link, so "
                                             "fewer cameras do not raise the frame rate"}
    result["plans"] = plans
    return result


MODES = ("fit", "full_res", "fewer_cameras")
MODE_TITLES = {
    "fit": "A  all cameras at the target frame rate, smaller image",
    "full_res": "B  all cameras at full resolution, lower frame rate",
    "fewer_cameras": "C  full resolution at the target frame rate, fewer cameras",
}


def recommend(report: Dict[str, Any], cfg: RecorderConfig, fps: float,
              bench: bool, mode: str = "fit") -> Dict[str, Any]:
    adapters = report["network"]["adapters"]
    cams = [c for c in report["cameras"].get("cameras", []) if not c.get("error")]
    rec: Dict[str, Any] = {"fps": fps, "groups": [], "warnings": [], "actions": []}
    warn, act = rec["warnings"].append, rec["actions"].append

    # --- cameras per host adapter
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for c in cams:
        a = adapter_for(c.get("ip", ""), adapters) if c.get("ip") else None
        c["_adapter"] = a
        groups.setdefault(a["name"] if a else "(unknown)", []).append(c)
    for name, members in groups.items():
        a = members[0]["_adapter"]
        rec["groups"].append(plan_group(members, a, fps))
    for c in cams:
        c.pop("_adapter", None)

    gige = [c for c in cams if "gige" in c.get("device_class", "").lower()]
    all_ptp = bool(cams) and all("ptp" in c.get("capabilities", []) and
                                 "action_command" in c.get("capabilities", []) for c in cams)
    rec["sync_mode"] = ("ptp_action" if all_ptp else
                        "hardware_trigger (wire the trigger inputs)" if cams else "auto")
    if len(groups) > 1 and all_ptp:
        warn("cameras are on different network adapters; PTP sync needs all cameras on "
             "one switch and one network")

    # --- cameras
    for c in cams:
        if c.get("link_speed_mbps") and int(c["link_speed_mbps"]) < 1000:
            warn(f"camera {c['serial']} links at only {c['link_speed_mbps']} Mb/s: check "
                 f"its cable and switch port")
        if c.get("ip", "").startswith("169.254."):
            act(f"give camera {c['serial']} a fixed IP (pylon IP Configurator) instead "
                f"of the automatic 169.254.x.x address")
            break

    # --- adapters
    used = {g["adapter"] for g in rec["groups"] if g.get("adapter")}
    link_local = [a["name"] for a in adapters
                  if any(ip.startswith("169.254.") for ip, _ in a.get("ipv4", []))]
    if len(link_local) > 1:
        warn(f"several adapters have 169.254.x.x addresses ({', '.join(link_local)}); "
             f"cameras may send to the wrong one. Give the camera adapter a fixed IP.")
    for a in adapters:
        if a["name"] not in used:
            continue
        if a.get("usb"):
            warn(f"{a['name']} is a USB network adapter ({a.get('driver') or a.get('description')}). "
                 f"These lose packets under sustained load without counting them; a PCIe "
                 f"card (Intel) is much more reliable for multi-camera streaming.")
        if a.get("wireless"):
            warn(f"{a['name']} is wireless; GigE cameras need a wired adapter")
        if (a.get("mtu") or 0) < 9000:
            if IS_LINUX:
                if (a.get("max_mtu") or 9000) >= 9000:
                    act(f"enable jumbo frames: sudo ip link set {a['name']} mtu 9000 "
                        f"(make it permanent in the network settings)")
                else:
                    warn(f"{a['name']} cannot do jumbo frames (max MTU {a.get('max_mtu')}); "
                         f"1500-byte packets work but cost more CPU")
            else:
                act(f"enable jumbo frames on '{a['name']}': Device Manager > adapter > "
                    f"Advanced > Jumbo Packet = 9014 (needs admin)")
        if IS_LINUX and a.get("rx_ring") and a.get("rx_ring_max") and \
                a["rx_ring"] < a["rx_ring_max"]:
            act(f"enlarge the receive ring: sudo ethtool -G {a['name']} rx {a['rx_ring_max']}")
        if IS_WINDOWS:
            if a.get("network_category") == "Public":
                warn(f"'{a['name']}' is classified as a Public network: the strictest "
                     f"firewall rules apply and usually block the camera stream")
            if not any("on" in b for b in a.get("basler_driver", [])):
                act(f"install/enable the Basler GigE Performance Driver on '{a['name']}' "
                    f"(pylon Camera Software Suite, needs admin): it receives the stream "
                    f"below the firewall and uses less CPU")
            if a.get("receive_buffers") and a.get("receive_buffers_max") and \
                    str(a["receive_buffers"]) != str(a["receive_buffers_max"]):
                act(f"raise Receive Buffers on '{a['name']}' to {a['receive_buffers_max']} "
                    f"(adapter Advanced settings, needs admin)")
    host = report["network"]["host"]
    if IS_LINUX and (host.get("rmem_max") or 0) < 32 * 2**20 and gige:
        act("raise the socket receive buffer: sudo sysctl -w net.core.rmem_max=33554432 "
            "(permanent: add net.core.rmem_max=33554432 to /etc/sysctl.conf)")
    if IS_WINDOWS and gige:
        on = [p.get("Name") for p in host.get("firewall", []) if p.get("Enabled")]
        if on:
            act(f"Windows Firewall is on ({', '.join(on)}): allow inbound UDP for "
                f"{sys.executable}, or use the Basler Performance Driver")

    # --- the three options, combined over all adapters
    rec["mode"] = mode
    options: Dict[str, Dict[str, Any]] = {}
    for m in MODES:
        plans = [g.get("plans", {}).get(m) for g in rec["groups"]]
        if not plans or any(p is None for p in plans):
            continue
        problems = [p["problem"] for p in plans if p.get("problem")]
        if problems:
            options[m] = {"problem": "; ".join(problems)}
            continue
        if m == "full_res":
            # One frame rate for everyone: the slowest group sets it.
            common = min(p["fps"] for p in plans)
            for p in plans:
                p["fps"] = common
        options[m] = {
            "fps": plans[0]["fps"],
            "cameras": sum(len(p["cameras"]) for p in plans),
            "of": len(cams),
            "size": f"{plans[0]['roi']['width']}x{plans[0]['roi']['height']}",
            "binning": plans[0].get("binning", 1),
            "coverage": plans[0].get("coverage", 1.0),
            "px_s": sum(len(p["cameras"]) * p["payload_bytes"] * p["fps"] for p in plans),
            "dropped": [d for p in plans for d in p.get("dropped", [])],
        }
    rec["options"] = options
    chosen = options.get(mode, {})
    if mode != "fit" and chosen.get("problem"):
        warn(f"option '{mode}' is not possible here: {chosen['problem']}")

    # --- encoder, sized for the chosen option
    n_enc = chosen.get("cameras") or max(1, len(cams))
    px_need = chosen.get("px_s")
    enc = report.get("ffmpeg", {}).get("encoders", [])
    works = {(e["codec"], e["hardware"]) for e in enc if e["works"]}
    gpu = next((hw for hw in ("nvidia", "intel", "amd", "vaapi")
                if (("h265", hw) in works or ("h264", hw) in works)), None)
    size = chosen.get("size") or next(
        (f"{g['roi']['width']}x{g['roi']['height']}" for g in rec["groups"] if g.get("roi")),
        None)
    if gpu:
        codec = "h265" if ("h265", gpu) in works else "h264"
        rec["encoder"] = {"codec": codec, "hardware": gpu,
                          "why": f"{codec.upper()} on the GPU works with {n_enc} "
                                 f"parallel encodes and leaves the CPU free"}
    else:
        rec["encoder"] = {"codec": "h264", "hardware": "cpu",
                          "why": "no working GPU encoder" if enc else
                          "encoders not tested (--quick); H.264 on the CPU as a safe default"}
        ffmpeg = report.get("ffmpeg", {}).get("path")
        if bench and ffmpeg and size and px_need:
            w, h = (int(x) for x in size.split("x"))
            cores = report["system"].get("physical_cores") or 1
            per_thread = benchmark_cpu_encoder(
                ffmpeg, w, h, ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"])
            if per_thread:
                # Pixels per second the CPU can encode, leaving room for grabbing.
                have = per_thread * w * h * cores * 0.6
                rec["cpu_h264_px_s"] = have
                if have < px_need:
                    mj = benchmark_cpu_encoder(ffmpeg, w, h, ["-c:v", "mjpeg", "-q:v", "3"])
                    if mj and mj * w * h * cores * 0.6 >= px_need:
                        rec["encoder"] = {
                            "codec": "mjpeg", "hardware": "cpu",
                            "why": (f"the CPU encodes about {have / 1e6:.0f} Mpixel/s of "
                                    f"H.264 but {px_need / 1e6:.0f} are needed; MJPEG is "
                                    f"cheap to encode (files about 10x larger)")}
                    else:
                        warn(f"the CPU encodes only about {have / 1e6:.0f} Mpixel/s of "
                             f"H.264 but this option needs {px_need / 1e6:.0f}: use a GPU "
                             f"encoder, or fewer pixels / a lower frame rate")

    # --- storage per option, and the disk
    bpp = BITS_PER_PIXEL[rec["encoder"]["codec"]]
    free = report.get("disk", {}).get("free_gb")
    for o in options.values():
        if o.get("px_s"):
            o["storage_gb_per_hour"] = round(o["px_s"] * bpp / 8 * 3600 / 1e9, 1)
            if free:
                o["hours_on_disk"] = round(free / o["storage_gb_per_hour"], 1) \
                    if o["storage_gb_per_hour"] else None
            have = rec.get("cpu_h264_px_s")
            if have and rec["encoder"]["hardware"] == "cpu" and \
                    rec["encoder"]["codec"] == "h264":
                o["cpu_ok"] = have >= o["px_s"]
    if chosen.get("storage_gb_per_hour"):
        rec["storage_gb_per_hour"] = chosen["storage_gb_per_hour"]
        rec["hours_on_disk"] = chosen.get("hours_on_disk")
        speed = report.get("disk", {}).get("write_mb_s")
        mb_s = chosen["storage_gb_per_hour"] * 1e3 / 3600
        if speed and mb_s * 3 > speed:
            warn(f"the disk writes {speed:.0f} MB/s; the recording needs about "
                 f"{mb_s:.0f} MB/s plus headroom")
    return rec


def recommended_config(base: RecorderConfig, rec: Dict[str, Any]) -> RecorderConfig:
    """The chosen option (rec['mode']) as a recorder config."""
    from syncrec.config import CameraOverride
    cfg = RecorderConfig.from_dict(base.to_dict())
    mode = rec.get("mode", "fit")
    plans = [g.get("plans", {}).get(mode) for g in rec["groups"]]
    plans = [p for p in plans if p and not p.get("problem")]
    if plans:
        p = plans[0]
        cfg.acquisition.fps = p["fps"]
        cfg.acquisition.roi.width = p["roi"]["width"]
        cfg.acquisition.roi.height = p["roi"]["height"]
        cfg.acquisition.roi.offset_x = p["roi"]["offset_x"]
        cfg.acquisition.roi.offset_y = p["roi"]["offset_y"]
        cfg.acquisition.binning = p.get("binning", 1)
        g = next(g for g in rec["groups"] if g.get("plans", {}).get(mode) is p)
        if not g.get("usb"):
            cfg.acquisition.packet_size = g["packet_size"]
            cfg.acquisition.inter_packet_delay_us = p["inter_packet_delay_us"]
        # Fewer cameras: switch the others off (they stay listed, so turning one
        # back on is a tick in the GUI).
        for serial in (d for p in plans for d in p.get("dropped", [])):
            ov = cfg.cameras.get(serial) or CameraOverride()
            ov.enabled = False
            cfg.cameras[serial] = ov
        for serial in (c for p in plans for c in p.get("cameras", [])):
            if serial in cfg.cameras:
                cfg.cameras[serial].enabled = True
    else:
        cfg.acquisition.fps = rec["fps"]
    cfg.acquisition.pixel_format = "Mono8"
    if rec.get("sync_mode", "").startswith("ptp_action"):
        cfg.sync.mode = "ptp_action"
    enc = rec.get("encoder") or {}
    if enc:
        cfg.encoder.codec = enc["codec"]
        cfg.encoder.hardware = enc["hardware"]
    cfg.session.segment_minutes = cfg.session.segment_minutes or 60
    if cfg.session.filename_pattern.endswith(".mp4"):
        cfg.session.filename_pattern = cfg.session.filename_pattern[:-4] + ".mkv"
    return cfg


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def print_options(rec: Dict[str, Any]) -> None:
    opts = rec.get("options", {})
    if not opts:
        return
    print("\n  Options:")
    for m in MODES:
        o = opts.get(m)
        if o is None:
            print(f"   {MODE_TITLES[m]}\n      not available: the full sensor size could "
                  f"not be measured on the cameras")
            continue
        mark = "  <- chosen (--mode)" if m == rec["mode"] else ""
        print(f"   {MODE_TITLES[m]}{mark}")
        if o.get("problem"):
            print(f"      not possible: {o['problem']}")
            continue
        view = (f"whole view, {o['binning']}x{o['binning']} binned" if o["binning"] > 1
                else "full sensor" if o["coverage"] >= 0.999
                else f"cut-out {o['coverage']:.0%} of the sensor")
        line = (f"      {o['size']} ({view}), {o['fps']:g} fps, "
                f"{o['cameras']} of {o['of']} cameras")
        if o.get("storage_gb_per_hour"):
            line += f", ~{o['storage_gb_per_hour']} GB/h"
        if o.get("cpu_ok") is False:
            line += "  (too much for the CPU encoder)"
        print(line)
    print("   (python -m syncrec check --mode full_res / fewer_cameras for the others)")


def print_report(report: Dict[str, Any], rec: Dict[str, Any]) -> None:
    def h(title: str) -> None:
        print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")

    s = report["system"]
    h("PC")
    print(f"  OS          {s['os']}")
    print(f"  CPU         {s.get('cpu')}  ({s.get('physical_cores')} cores / "
          f"{s.get('logical_cpus')} threads)")
    print(f"  RAM         {s.get('ram_gb')} GB")
    print(f"  Python      {s['python']}   pypylon {s.get('pypylon') or 'NOT INSTALLED'}   "
          f"PySide6 {s.get('PySide6') or '-'}")
    for g in report["gpus"] or [{"name": "none found"}]:
        print(f"  GPU         {g.get('name')}  {g.get('driver') or ''}")

    h("Network adapters")
    for a in report["network"]["adapters"]:
        ips = ", ".join(f"{ip}/{p}" for ip, p in a.get("ipv4", [])) or "no IPv4"
        kind = "USB" if a.get("usb") else ("wireless" if a.get("wireless") else "PCIe/onboard")
        speed = f"{a['speed_mbps']} Mb/s" if a.get("speed_mbps") else "no link"
        print(f"  {a['name']:<14} {speed:<12} MTU {a.get('mtu') or '?':<6} {kind:<13} "
              f"{a.get('driver') or a.get('description', '')}")
        print(f"  {'':<14} {ips}")
        extra = []
        if a.get("max_mtu"):
            extra.append(f"max MTU {a['max_mtu']}")
        if a.get("rx_ring"):
            extra.append(f"RX ring {a['rx_ring']}/{a.get('rx_ring_max')}")
        if a.get("jumbo"):
            extra.append(f"Jumbo Packet {a['jumbo']}")
        if a.get("receive_buffers"):
            extra.append(f"Receive Buffers {a['receive_buffers']}")
        if a.get("network_category"):
            extra.append(f"network {a['network_category']}")
        if a.get("basler_driver"):
            extra.append("Basler driver: " + ", ".join(a["basler_driver"]))
        if extra:
            print(f"  {'':<14} " + " | ".join(extra))
    host = report["network"]["host"]
    if "rmem_max" in host:
        print(f"  socket receive buffer limit (rmem_max): {host['rmem_max']}")
    for p in host.get("firewall", []):
        print(f"  firewall {p.get('Name')}: {'on' if p.get('Enabled') else 'off'}")

    h("Cameras")
    cams = report["cameras"]
    if cams.get("error"):
        print(f"  {cams['error']}")
    for c in cams.get("cameras", []):
        print(f"  {c['serial']}  {c['model']}  {c.get('device_class')}  {c.get('ip') or ''}"
              f"  firmware {c.get('firmware', '?')}")
        if c.get("error"):
            print(f"      {c['error']}")
            continue
        print(f"      sensor {c.get('width_max')}x{c.get('height_max')}  link "
              f"{c.get('link_speed_mbps', '?')} Mb/s  packet {c.get('packet_size', '?')}  "
              f"PTP {(c.get('ptp') or {}).get('state', '-')}  action queue "
              f"{c.get('action_queue_size') or '?'}")
        print(f"      capabilities: {', '.join(c.get('capabilities', []))}")
        lims = c.get("frame_rate_limits", [])
        if lims:
            print("      camera's own max fps: " + ", ".join(
                f"{l['width']}x{l['height']} {l['camera_max_fps']}" for l in lims))

    ff = report.get("ffmpeg", {})
    h("ffmpeg and encoders")
    print(f"  {ff.get('path') or ff.get('error')}  {ff.get('version', '')}")
    for e in ff.get("encoders", []):
        print(f"  [{' OK ' if e['works'] else ' -- '}] {e['codec']:<6} {e['hardware']:<7} "
              f"{e['detail']}")

    d = report.get("disk", {})
    h("Recording disk")
    print(f"  {d.get('path')}: {d.get('free_gb')} GB free of {d.get('total_gb')} GB"
          + (f", writes {d['write_mb_s']:.0f} MB/s" if d.get("write_mb_s") else ""))

    h(f"RECOMMENDATION for {rec['fps']:g} fps")
    if not rec["groups"]:
        print("  No cameras could be read, so no image settings can be recommended.")
    for g in rec["groups"]:
        if g.get("usb"):
            print(f"  {len(g['cameras'])} USB camera(s). {g['note']}")
        else:
            print(f"  {len(g['cameras'])} camera(s) on adapter "
                  f"{g['adapter'] or '(not found)'}: host link {g['host_link_mbps']} Mb/s, "
                  f"camera links {g['camera_link_mbps']} Mb/s, jumbo frames "
                  f"{'yes' if g['jumbo'] else 'no'}")
        print(f"  budget per camera: {g['per_camera_budget_mb_s']} MB/s\n")
        print(f"  {'image size':<14}{'view':<22}{'max fps':>8}{'(link':>8}{'camera)':>9}")
        for t in g["size_table"]:
            mark = "  <- option A" if g.get("roi") and \
                (t["width"], t["height"], t["binning"]) == \
                (g["roi"]["width"], g["roi"]["height"], g.get("binning", 1)) else ""
            view = (f"full, {t['binning']}x{t['binning']} "
                    f"{'decimated' if t.get('decimation') else 'binned'}"
                    if t["binning"] > 1 else
                    "full sensor" if t["coverage"] >= 0.999 else
                    f"cut-out {t['coverage']:.0%}")
            print(f"  {t['width']:>5} x {t['height']:<6}{view:<22}{t['max_fps']:>8}"
                  f"{t['max_fps_bandwidth']:>8}{t['max_fps_camera'] or '-':>9}{mark}")
    print_options(rec)
    for g in rec["groups"]:
        p = g.get("plans", {}).get(rec["mode"])
        if not p or p.get("problem"):
            if g.get("problem") and rec["mode"] == "fit":
                print(f"\n  ! {g['problem']}: lower the frame rate, or use a faster link")
            continue
        r = p["roi"]
        title = MODE_TITLES[rec["mode"]][3:]
        print(f"\n  Settings for option {MODE_TITLES[rec['mode']][0]} ({title}):")
        print(f"  frame rate            {p['fps']:g} fps")
        if rec["mode"] == "fewer_cameras":
            print(f"  cameras               {len(p['cameras'])} of "
                  f"{len(p['cameras']) + len(p.get('dropped', []))}: "
                  f"{', '.join(p['cameras'])}")
            if p.get("dropped"):
                print(f"  switched off          {', '.join(p['dropped'])} (any "
                      f"{len(p['cameras'])} will do; choose in the GUI)")
        if p.get("binning", 1) > 1:
            print(f"  image size            {r['width']} x {r['height']}, binning "
                  f"{p['binning']} (whole field of view)")
        else:
            print(f"  image size            {r['width']} x {r['height']} "
                  f"(offset {r['offset_x']}, {r['offset_y']}"
                  f"{' = centred' if p.get('coverage', 1.0) < 0.999 else ' = full sensor'})")
            if p.get("coverage", 1.0) < 0.999:
                full = next((t for t in g["size_table"]
                             if t["binning"] == 1 and t["coverage"] >= 0.999), None)
                can_bin = any(t["binning"] > 1 for t in g["size_table"])
                print(f"  ! this is a CUT-OUT: {p['coverage']:.0%} of the sensor area"
                      + ("" if can_bin else " (these cameras cannot bin)")
                      + f". The whole view at {rec['fps']:g} fps needs more bandwidth"
                      + (f"; whole view on this link: up to {full['max_fps']} fps"
                         if full else "") + ".")
        if g.get("usb"):
            continue
        print(f"  packet size           {g['packet_size']}")
        print(f"  inter-packet delay    {p['inter_packet_delay_us']} us   "
              f"(one frame takes {p['frame_transfer_ms']} ms to send of "
              f"{1000 / p['fps']:.0f} ms; link use {p['average_link_share']:.0%} average, "
              f"{p['burst_link_share']:.0%} while all send)")
    print(f"  sync mode             {rec['sync_mode']}")
    e = rec.get("encoder", {})
    print(f"  encoder               {e.get('codec')} on {e.get('hardware')}  "
          f"({e.get('why', '')})")
    if "storage_gb_per_hour" in rec:
        print(f"  storage               about {rec['storage_gb_per_hour']} GB per hour "
              f"(rough; depends on the scene)"
              + (f", about {rec['hours_on_disk']} h on the free space"
                 if rec.get("hours_on_disk") else ""))
    print("  long recordings       segments of 60 min, .mkv files")
    if rec["actions"]:
        print("\n  Recommended changes to this PC:")
        for a in rec["actions"]:
            print(f"   - {a}")
    if rec["warnings"]:
        print("\n  Warnings:")
        for w in rec["warnings"]:
            print(f"   ! {w}")
    print("\n  Verify with a test recording: the log must not say 'cannot keep up', and\n"
          "  the Missed / Dropped / Filled columns should stay at or near 0.")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="syncrec check",description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fps", type=float, default=None,
                    help="target frame rate (default: from the config, else 25)")
    ap.add_argument("-c", "--config", help="recorder config to start from")
    ap.add_argument("--write-config", metavar="PATH",
                    help="save the recommended settings as a recorder config")
    ap.add_argument("--json", metavar="PATH", help="also save the full report as JSON")
    ap.add_argument("--quick", action="store_true",
                    help="skip the encoder tests, disk speed test and CPU benchmark")
    ap.add_argument("--mode", choices=MODES, default="fit",
                    help="which option to detail and to write with --write-config: "
                         "fit = all cameras at the target fps with a smaller image "
                         "(default); full_res = all cameras at full resolution, lower "
                         "fps; fewer_cameras = full resolution at the target fps with "
                         "as many cameras as the link allows")
    ap.add_argument("--prefer-pixels", action="store_true",
                    help="recommend the largest image even if it is a cut-out "
                         "(default: the whole field of view, binned if necessary)")
    ap.add_argument("--no-camera-measure", action="store_true",
                    help="do not change camera settings to measure frame-rate limits")
    args = ap.parse_args(argv)
    global PREFER_FULL_VIEW
    PREFER_FULL_VIEW = not args.prefer_pixels

    cfg = RecorderConfig.load(args.config) if args.config else RecorderConfig()
    fps = args.fps or (cfg.acquisition.fps if args.config else 25.0)

    steps = [("PC", lambda: collect_system()), ("GPUs", collect_gpus),
             ("network", collect_network),
             ("cameras", lambda: collect_cameras(cfg, not args.no_camera_measure))]
    report: Dict[str, Any] = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                              "platform": sys.platform}
    keys = ["system", "gpus", "network", "cameras"]
    for (label, fn), key in zip(steps, keys):
        print(f"reading {label}...", flush=True)
        try:
            report[key] = fn()
        except Exception as exc:
            report[key] = {"error": str(exc)} if key != "gpus" else []
            print(f"  failed: {exc}")
    n_cams = len([c for c in report["cameras"].get("cameras", []) if not c.get("error")])
    print("testing ffmpeg encoders..." if not args.quick else "reading ffmpeg...", flush=True)
    report["ffmpeg"] = collect_ffmpeg(cfg, max(1, n_cams), not args.quick)
    print("checking the recording disk...", flush=True)
    try:
        report["disk"] = collect_disk(cfg.session.output_dir, not args.quick)
    except OSError as exc:
        report["disk"] = {"error": str(exc)}
    if not args.quick:
        print("measuring the CPU encoder...", flush=True)
    rec = recommend(report, cfg, fps, bench=not args.quick, mode=args.mode)
    report["recommendation"] = rec

    print_report(report, rec)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"\nfull report saved to {args.json}")
    if args.write_config:
        option = rec.get("options", {}).get(args.mode)
        if not option or option.get("problem"):
            why = (option or {}).get("problem") or "no camera measurements for it"
            print(f"\nNOT writing {args.write_config}: option '{args.mode}' is not "
                  f"available ({why}).")
            return 1
        recommended_config(cfg, rec).save(args.write_config)
        print(f"recommended settings ({MODE_TITLES[args.mode][3:]}) saved to "
              f"{args.write_config} (use: python -m syncrec gui -c {args.write_config})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
