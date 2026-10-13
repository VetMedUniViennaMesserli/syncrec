# Synchronized Multi-Camera Recorder

Standalone, frame-synchronised video recorder for any number of Basler cameras.
Runs on its own machine, writes one video per camera named `<clip>_<serial>.mp4` plus
the per-frame timing data that proves the cameras really were in sync — which is what
any multi-view analysis (3D tracking, triangulation, ...) depends on.

---

## Installation

### Linux: installer

Each [release](../../releases) has a package for Ubuntu 22.04 or newer and Debian 12
or newer:

```bash
sudo apt install ./syncrec_<version>_all.deb
```

apt also installs Python, ffmpeg and the libraries the GUI needs. **Installing needs
internet access**: the package then downloads the Python packages the recorder uses
(pypylon, NumPy, PySide6) from [PyPI](https://pypi.org) into its own environment in
`/opt/syncrec/venv`; they are not part of the package.

The recorder then appears in the applications menu as *Synchronized Multi-Camera
Recorder*, and `syncrec` works in a terminal wherever `python -m syncrec` is used
below (`syncrec --help`). Installed this way, **Save as default** writes
`~/.config/syncrec/settings.json`, so every user account has its own defaults.

`sudo dpkg-reconfigure syncrec` downloads the Python packages again (e.g. after a
distribution upgrade changed the Python version). `sudo apt remove syncrec` removes
everything except the users' settings.

### From source: Linux and Windows

* **Python 3.9 or newer.**
* **ffmpeg**, which does the video encoding. It must be on the `PATH`, or set its
  location in `encoder.ffmpeg` (GUI: advanced settings, *ffmpeg*). Download it from
  [ffmpeg.org](https://ffmpeg.org/download.html) or use your package manager.
  `python -m syncrec encoders` shows whether it is found and which encoders work.
* **The Python packages** in `requirements.txt`: pypylon (Basler's camera SDK, which
  includes the pylon runtime and the camera emulator), NumPy, and PySide6 for the GUI.

From the repository folder:

```bash
python -m pip install -r requirements.txt
```

Run from source, **Save as default** writes `settings.json` in the repository folder.

You also need **Basler cameras** (GigE or USB3). Without cameras, see
[Working without cameras](#working-without-cameras).

---

## Quick start

```bash
python -m syncrec                        # GUI
python -m syncrec list                   # what cameras can I see?
python -m syncrec record --duration 60   # headless, 60 seconds
```

(With the Linux installer: `syncrec`, `syncrec list`, ...)

### Using the GUI

The window opens with the settings needed for everyday recording: which cameras to
use, the folder, the recording name, frame rate, exposure and gain. Live images
appear as soon as the cameras are found, so exposure and gain can be adjusted while
watching the result. The live preview is **not synchronised** and saves nothing;
synchronisation is set up when **Record** is pressed. If the sync mode is
`ptp_action`, PTP is switched on during the preview, so the clocks are usually
locked by the time Record is pressed.

**Settings → Show advanced settings** (Ctrl+Shift+A) shows everything else: sync
mode, resolution, binning, pixel format, codec and encoder, duration, file names,
segments, and the config buttons. **Save as default** writes the current settings to
`settings.json`, which is loaded every time the program starts. Changes made in the
basic view are not saved, so the next session starts from the defaults again.

**Recording name**: names the recording's folder and (via `{clip}`) its files.
`{date}` and `{time}` are filled in when Record is pressed, e.g.
`mouse3_{date}_{time}` → `mouse3_20261013_141500`. The default name comes from
`session.clip_name` in `settings.json`. A line under the field shows exactly where the
files will go, and **?** explains the variables. Nothing is overwritten: if a
recording with that name already exists, `_2`, `_3`, ... is added.

No cameras yet? Everything below works without hardware — see
[Working without cameras](#working-without-cameras).

---

## Synchronising the cameras

This is the part that matters, and the right answer depends on how the cameras are
wired.
Four modes are implemented; pick one with `sync.mode` in the config or the **Mode**
dropdown in the GUI.

| Mode | Accuracy | Needs | Use when |
|---|---|---|---|
| `ptp_action` | **< 100 µs** | GigE cameras on one switch | GigE cameras with no trigger cabling |
| `hardware_trigger` | **< 10 µs** | A trigger wire to every camera | Best accuracy, and the only option on USB3 |
| `software_trigger` | ~0.1–5 ms | Nothing | Bench testing; too loose for fast motion |
| `freerun` | **not synchronised** | Nothing | Fallback only; frames must be aligned afterwards |
| `auto` | — | — | Picks `ptp_action`, else `software_trigger`, else `freerun` |

`auto` never selects `hardware_trigger`: software cannot tell whether the trigger
cable is actually plugged in, and guessing wrong records nothing at all. Select it
explicitly once the trigger wiring is in place.

### `ptp_action` — PTP + scheduled action commands (recommended for GigE)

The cameras discipline their clocks to each other over IEEE 1588 (PTP); one elects
itself master and the rest slave to it. The host then broadcasts *scheduled* action
commands meaning "everyone expose at time T". Because T is agreed in advance, jitter
in the host, the OS and the network does not affect sync at all — accuracy is just
the PTP residual, typically tens of microseconds.

Requirements: all cameras on the **same switch** (PTP degrades badly across routers),
GigE only (USB3 cameras have no PTP), and PTP traffic must not be filtered. The
recorder waits for one master + N-1 slaves with offsets inside `sync.ptp.max_offset_ns`
(for `settle_s` seconds) before it will start, and fails with a clear message if
that never happens.

During the recording it keeps watch every 10 s: it re-measures the camera clock
against the PC clock (they drift apart by seconds per day, and the commands must
keep arriving one lead time early), follows the schedule if the PTP time base jumps
(e.g. a camera restarted and became master), and logs a warning if PTP roles or
offsets degrade. Frame *n* of every video is trigger *n*: missed triggers are filled
with the previous image and marked as placeholders in the CSV
(`session.fill_missing_frames`).

### `hardware_trigger` — one wire, every camera

The most accurate option, and the only synchronised mode available on USB3. Two
variants:

* **External generator** — leave `sync.hardware.master_serial` empty. Every camera is
  armed on `trigger_source` (default `Line1`) and waits. Feed them your signal at the
  frame rate you want; the recorder's `fps` setting is then only a buffer-sizing hint.
* **One camera as master** — set `master_serial`. That camera free-runs at `fps` and
  drives `master_line` (default `Line2`) from `master_line_source` (default
  `ExposureActive`); the others are slaved to it. No extra hardware, but the master's
  own exposure defines the timing.

Check the trigger line numbering against your camera's datasheet — it differs between
ace, ace 2 and dart.

### `software_trigger`

The host tells each camera to expose, one call after the next. Spread is the sum of
per-camera command latency. Honest about what it is: fine for testing the pipeline,
not for fast motion.

### `freerun`

No trigger at all. Cameras run on their own oscillators and drift apart by
milliseconds within seconds. Recorded anyway, with full timestamps, so you *can*
align afterwards — but treat this as a fallback, not a sync mode.

---

## Working without cameras

Two independent ways to run the whole system with no hardware:

**1. The pylon camera emulator** (exercises the real pypylon code path):

```bash
PYLON_CAMEMU=8 python -m syncrec          # Windows cmd: set PYLON_CAMEMU=8, then run
```

Eight emulated Basler devices (or however many you ask for) appear, open, configure and grab like real ones. They
do not support PTP, action commands or device timestamps, so only `software_trigger`
and `freerun` are available.

**2. The built-in dummy cameras** (`--backend dummy`):

Full simulation of the things the emulator cannot do — device clocks with drift, PTP
convergence, action commands, frame drops. They render a moving object (four markers
on a rigid body) in a virtual room, one known camera pose per dummy (`dummy.count`), with
the frame number and timestamp burned into every image. That makes de-sync visible by
eye, and gives a dummy recording known 3D ground truth to test analysis code against.
Exposure and gain change the dummy images' brightness, so the live preview reacts.

This is the only way to test the sync logic itself without hardware:

```bash
python tools/selftest.py
```

records four seconds in each sync mode and asserts the synchronised ones land inside
1 ms while free run does not.

---

## Configuration

One JSON file holds everything. `settings.json` (next to the `syncrec` folder, or
`~/.config/syncrec/` with the Linux installer) is loaded automatically by the GUI and the command line, wherever they are started
from; `-c other.json` uses another file instead. Without `settings.json` the built-in
defaults (in `syncrec/config.py`) apply. `settings.json` belongs to one recording
machine, so it is not part of the repository (see `.gitignore`). `config.example.json` shows per-camera
overrides. Write a file with the built-in defaults with:

```bash
python -m syncrec config --write my_config.json
```

In the GUI, **Save as default** (advanced settings) writes whatever is on screen to
`settings.json`, creating it the first time.

Keys worth knowing:

| Key | Meaning |
|---|---|
| `session.clip_name` | Default recording name; `{date}` and `{time}` are filled in |
| `acquisition.backend` | `auto` (Basler if present, else dummy), `pylon`, `dummy` |
| `acquisition.roi` | Sensor ROI; smaller ROI is the cheapest way to raise the frame rate |
| `acquisition.queue_size` | Frames buffered per camera before dropping. Costs RAM: frame size × cameras × depth |
| `sync.max_spread_ms` | Spread above which the UI turns red and the report flags the frame |
| `encoder.output_args` | ffmpeg arguments, passed through verbatim |
| `cameras.<serial>.enabled` | Leave a camera out without unplugging it |
| `cameras.<serial>.rotate_180` | Sets `ReverseX`+`ReverseY` **in the camera** |
| `cameras.<serial>.exposure_us` | Per-camera exposure override |

> `rotate_180` changes what the sensor delivers, so recordings made with it no longer
> match a camera calibration computed without it (and vice versa). Decide before
> calibrating, and keep it the same afterwards.

### Codecs and GPU encoding

Pick a codec and where to encode it; the recorder generates the ffmpeg arguments
(GUI: *Codec*, *Encode on*, *Quality*; config: `encoder.codec`, `encoder.hardware`,
`encoder.quality`; command line: `--video-codec`, `--hw`, `--quality`).

| Codec | CPU | NVIDIA (NVENC) | Intel (Quick Sync) | AMD (AMF, Windows) | VAAPI (Linux) |
|---|---|---|---|---|---|
| `h264` | libx264 | h264_nvenc | h264_qsv | h264_amf | h264_vaapi |
| `h265` | libx265 | hevc_nvenc | hevc_qsv | hevc_amf | hevc_vaapi |
| `av1` | libsvtav1 | av1_nvenc | av1_qsv | av1_amf | av1_vaapi |
| `mjpeg`, `ffv1` | CPU only | | | | |

`hardware: "auto"` tries the GPUs in that order and falls back to the CPU. Every
choice is tested before recording, with one encode per camera running at once:
ffmpeg lists GPU encoders even without a suitable GPU, AV1 needs a recent GPU
(NVIDIA RTX 40xx, Intel Arc, AMD RX 7000 or newer), and consumer GPUs limit how many
encodes run in parallel. An explicitly chosen encoder that does not work stops the
recording before it starts. What was used is in the log and in `manifest.json`.

See what works on a PC:

```bash
python -m syncrec encoders --sessions 8    # 8 parallel encodes per GPU (one per camera)
```

`quality` is lower = better and bigger; by default H.264 20, H.265 24, AV1 32
(roughly the same look). H.265 and AV1 give the same quality in about half and a
third less space than H.264, but cost several times more CPU: with many cameras,
use them on a GPU. `codec: "custom"` (or `--codec="..."`) passes `output_args` to
ffmpeg verbatim.

For scale: eight cameras at 2048×1536 mono @ 25 fps is ~630 MB/s of raw pixels. The recorder
tells you when encoding cannot keep up rather than silently losing frames: dropped
counts appear per camera in the GUI, in the summary, and in the metadata CSV.

---

## Output

```
recordings/20260925_143012/
    20260925_143012_<serial>.mp4     one video per camera, named by serial
    20260925_143012_<serial>.csv     per-frame timing for that camera
    ...
    manifest.json                    config, camera list, counts, sync report
```

The CSV has one row per grabbed frame:

| Column | Meaning |
|---|---|
| `frame_index` | Our own per-camera counter |
| `block_id` | The camera's frame counter; gaps here mean the camera lost frames |
| `trigger_id` | Which trigger this frame answers (triggered modes only) |
| `cam_timestamp_ns` | Device clock — PTP-disciplined and directly comparable across cameras |
| `host_timestamp_ns` | When the host received it |
| `skipped_before` | Frames the camera reports as lost before this one |
| `queued_frames` | Encoder backlog at that moment |
| `written` | 0 if the frame was dropped rather than encoded |

### Long recordings: segments

For recordings over hours or days, set *Split every* in the GUI (or
`session.segment_minutes`, or `--segment-minutes 60`). The recording is then split
into folders `segment_0000/`, `segment_0001/`, ... next to `manifest.json`, each
holding one video and one per-frame CSV per camera, named as usual.

The split happens live, inside the running encoder: ffmpeg starts a new file at an
exact frame count (a key frame is forced there), so there is no restart, no gap and
no duplicated frame, and all cameras split at the same trigger (segment *k* holds
triggers *k·N* to *k·N+N−1*). Each finished segment is checked about a minute later
and again at the end: every file must contain exactly *N* frames; any difference is
reported as an error and listed under `segments.problems` in `manifest.json`.

The viewer opens the recording folder or any segment, and stepping and playing run
on across segment boundaries (PageUp/PageDown jump between segments).
`tools/check_sync.py` and `tools/light_sync.py` take the recording folder (all
segments, frame numbers counted over the whole recording) or `--segment N`.

For multi-day recordings consider `.mkv` in the file name pattern
(`{clip}_{serial}.mkv`): an MP4 is only readable once it is closed, so a power cut
loses the segment being written, while an MKV stays readable up to the cut. The
recorder also stops cleanly when free disk space falls below `session.min_free_gb`
(default 5 GB), rather than letting the files break on a full disk.

### Verifying a recording

```bash
python tools/check_sync.py recordings/20260925_143012 --tolerance-ms 1.0
```

Per camera: real frames, filled (placeholders), missed, dropped, the real frame
rate, frame interval and jitter. Then how many frame groups are complete and the
cross-camera spread (mean / median / p95 / max), and two verdicts: SYNC (were the
frames every camera recorded exposed together?) and COMPLETENESS (how many
camera-frames are placeholders). Exits non-zero if SYNC fails, so it can gate an
automated workflow. Split recordings are read one segment at a time (so weeks of
recording fit in memory), with a line per segment that has problems; `--segment N`
checks one. `--probe` also counts the frames in the video files with ffprobe and
compares them with the CSVs.

To check sync with your own eyes, switch a light off during a recording and run
`tools/light_sync.py <recording folder> --plot`: the first dark frame must have the
same number in every video.

### Viewing a recording

```bash
python -m syncrec view recordings/20260925_143012
```

Leave out the folder to pick one in a dialog. Plays all videos of a recording side by side, always
at the same frame number, with frame-by-frame stepping: Left/Right one frame,
Shift ten, Ctrl a hundred, Space play/pause, Home/End. The mouse wheel zooms a
camera (drag to pan, "Link zoom" to move all together, 0 to reset); double-click
shows one camera large. Each tile shows the trigger number and the exposure time
relative to the other cameras from the CSV, and a red border marks placeholder
frames. "Save frame" writes the current image of every camera as PNG into
`frames/` inside the recording folder.

---

## Using the recordings

With a triggered sync mode, frame *n* of every camera's video was exposed at the same
moment (missed frames are filled, see `session.fill_missing_frames`), so analysis
code can simply index all videos by frame number. The per-frame CSVs hold the
timestamps that prove it.

---

## Command line

```
python -m syncrec [list|record|config|gui|view|encoders|check] [options]

  --backend {auto,pylon,dummy}
  --sync {auto,ptp_action,hardware_trigger,software_trigger,freerun}
  --fps FPS              --duration SECONDS      (0 = until Ctrl+C)
  --output DIR           --clip NAME
  --codec="ARGS"         ffmpeg output args. The equals sign is required,
                         because the value itself starts with a dash.
  -c CONFIG (default: settings.json)             -v (verbose)
```

Options work before or after the subcommand. `record` stops cleanly on Ctrl+C:
triggering stops, the queues drain, ffmpeg is flushed and the manifest is written.

---

## How it is put together

```
syncrec/
  camera.py       CameraBackend interface -- everything above talks only to this
  backends/
    pylon_backend.py  real Basler cameras and the pylon emulator
    dummy_backend.py  simulated cameras, including simulated PTP and drops
    scene.py          the virtual scene the dummies render
  sync.py         the four sync strategies
  writer.py       one ffmpeg process per camera, plus the metadata CSV
  session.py      orchestrates cameras, threads, queues and writers
  live.py         free-running live preview before / between recordings
  cli.py          headless entry point
  ui/             PySide6 desktop UI
tools/
  check_sync.py   verify a finished recording
  selftest.py     record in every sync mode with dummies and check the result
packaging/linux/  builds the Linux installer (build.sh) and tests it (test_package.sh)
.github/workflows/linux-installer.yml
                  runs both when a release is published and attaches the files
```

Each camera gets two threads: one that does nothing but pull frames out of the driver,
and one that pipes them to ffmpeg, with a bounded queue between them. If encoding
hiccups the queue absorbs it and the camera keeps streaming; if the queue fills,
frames are dropped **deliberately and counted**, which is much better than stalling
the driver and losing frames without knowing.

Adding a different camera make means implementing `CameraBackend` and registering it
in `backends/discover()`. Nothing else needs to change.

---

## Checking a PC: recommended settings

```bash
python -m syncrec check
python -m syncrec check --fps 25 -c my_config.json --write-config recommended.json
```

Reads the PC (CPU, memory, GPUs), every network adapter (link speed, MTU/jumbo
frames, USB or PCIe, driver, receive buffers; firewall and network profile on
Windows; socket buffer limit on Linux), the recording disk (free space, write
speed), ffmpeg and which encoders really work (one encode per camera at once), and
every camera (model, firmware, IP, sensor, link speed, PTP/action support, and the
frame rate the camera itself reaches at several image sizes -- measured briefly,
settings restored afterwards). It then recommends image size, packet size,
inter-packet delay, sync mode and encoder for the target frame rate, lists the OS
changes worth making, estimates storage per hour, and with `--write-config` saves
it all as a config for the recorder. `--quick` skips the slower tests; `--json`
saves the full report. Close the recorder and pylon Viewer first.

It always shows three options, and `--mode` picks the one that is detailed and
written with `--write-config`:

| `--mode` | |
|---|---|
| `fit` (default) | all cameras at the target frame rate, image made smaller (binned if the cameras can, else a centred cut-out) |
| `full_res` | all cameras at full resolution, at the highest frame rate the link and cameras allow |
| `fewer_cameras` | full resolution at the target frame rate, with as many cameras as the link carries; the others are switched off in the config (any of them can be swapped in the GUI) |

Note that `acquisition.roi` **crops**: a smaller width/height is a cut-out of the
sensor, not a scaled-down image. To keep the whole field of view with less data,
use `acquisition.binning` (2 = 2x2 pixels combined in the camera, a quarter of the
data; width/height are then in binned pixels). Most Basler colour models cannot
bin; the recorder then uses decimation if the camera has it, and otherwise refuses
to start instead of silently cropping. The check prefers the whole field of view
and marks cut-outs clearly (`--prefer-pixels` for the largest image instead).

## Troubleshooting

Start with `python tools/diagnose_pylon.py`: it tests free run, software trigger,
action commands and (`--ptp`) scheduled triggering one step at a time and says which
step fails.

**No cameras found.** GigE cameras need an IP on the host's subnet -- use the pylon
IP Configurator. Check they are not open in pylon Viewer; a camera can only be
opened once.

**Cameras found, but no image at all (even in free run).** The image stream is
blocked on its way into the PC while control traffic works. On Windows this is
almost always the firewall (the camera network is usually classified "Public"):
allow inbound UDP for python.exe, or install the Basler GigE Performance Driver,
which receives below the firewall. Also check that only one network adapter has a
169.254.x.x address.

**PTP never converges.** All cameras must be on one switch. Confirm each reports
`PtpEnable`/`GevIEEE1588` in `list`. The first lock after power-up can take minutes
at a tight `max_offset_ns`; raise `sync.ptp.sync_timeout_s` if needed. With
`keep_enabled` the lock is kept between recordings.

**Triggered mode records nothing.** The cameras are armed and waiting for a trigger
that is not arriving. For `ptp_action` the first trigger is sent with
acknowledgements and the log says which cameras did not answer; for
`hardware_trigger` check the wiring and that `trigger_source` matches the line used.

**Only every 2nd or 3rd frame is new (Filled about 2x Written, Missed and Dropped
0).** The cameras ignore triggers. Common causes: (1) too many scheduled action
commands pending at once -- a camera holds only a few (ace classic: one); the
recorder limits the lead time to the camera's action queue and spot-checks
acknowledgements, and `sync.ptp.action_lead_time_ms` can be lowered further; (2) exposure too long for the frame period;
(3) frame transfer slower than the frame period (inter-packet delay). The log
names the cause at the start of the recording.

**Columns in the recording table.** *Missed*: lost between camera and PC (network:
incomplete or never arrived). *Dropped*: arrived, but discarded because encoding fell
behind (PC). *Filled*: video slots holding the previous image -- missed, dropped, or
triggers the camera ignored because it was not ready. Filled can be high while
Missed and Dropped are zero: then the camera cannot keep up with the trigger rate
(exposure too long, or frame transfer too slow because of the inter-packet delay).
The log says so at the start of a recording ("cannot keep up with ... fps").

**Missed frames ("grab failed: incompletely grabbed").** Packets are lost: too much
data for the link, switch buffers overflowing when all cameras fire at once, or a
network adapter that cannot cope. Set `acquisition.inter_packet_delay_us` so the
cameras share the link (the transfer of one frame must still fit in one frame
period), enable jumbo frames (MTU 9000, `packet_size` 8192), and on Linux raise
`net.core.rmem_max`. USB network adapters (driver r8152 and similar) lose packets
under sustained load without counting them; use a PCIe card (Intel) instead.

**Dropped frames climbing.** The encoder cannot keep up. Use a GPU encoder
(`encoder.hardware`), H.264 instead of H.265/AV1 on the CPU, lower the resolution or
frame rate, or write to a faster disk. Raising `queue_size` only buys time.

---

## License

MIT -- see [LICENSE](LICENSE).
