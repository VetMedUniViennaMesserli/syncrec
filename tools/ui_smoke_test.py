"""Drive the GUI headlessly: discover, live preview, record, stop, and save screenshots.

Useful on a machine with no display, and as a regression test for the UI wiring --
it exercises the same code paths the operator's clicks do.

    python tools/ui_smoke_test.py --out uishots
    python tools/ui_smoke_test.py --backend pylon --seconds 10 --show
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", help="where to write screenshots and the recording")
    parser.add_argument("--backend", default="dummy", choices=["auto", "pylon", "dummy"])
    parser.add_argument("--sync", default="ptp_action")
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--show", action="store_true",
                        help="use a real window instead of the offscreen platform")
    args = parser.parse_args(argv)

    if not args.show:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    out = args.out or tempfile.mkdtemp(prefix="syncrec_ui_")
    os.makedirs(out, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from syncrec.config import RecorderConfig
    from syncrec.ui.app import MainWindow, apply_dark_palette

    # A config of our own, so the test neither depends on nor changes settings.json.
    cfg = RecorderConfig()
    cfg.acquisition.backend = args.backend
    cfg.acquisition.roi.width, cfg.acquisition.roi.height = 640, 480
    cfg.acquisition.fps = 25
    cfg.sync.mode = args.sync
    cfg.sync.ptp.settle_s = 0.5               # simulated PTP; keep the test short
    cfg.session.output_dir = os.path.join(out, "recording")
    cfg.session.clip_name = "ui_smoke_{date}"
    cfg.encoder.codec = "custom"
    cfg.encoder.output_args = "-c:v libx264 -preset ultrafast -crf 28 -pix_fmt yuv420p".split()
    config_path = os.path.join(out, "ui_smoke_config.json")
    cfg.save(config_path)

    app = QApplication(sys.argv[:1])
    apply_dark_palette(app)
    window = MainWindow(config_path)
    window.resize(1500, 950)
    window.show()

    state = {"failures": []}

    def fail(message):
        print(f"FAIL: {message}")
        state["failures"].append(message)

    def shot(name):
        path = os.path.join(out, name)
        window.grab().save(path)
        print(f"  screenshot {name}")

    def after_discover():
        print(f"discovered {len(window.cameras)} cameras "
              f"via {window._backend_name}")
        if not window.cameras:
            fail("no cameras discovered")
            return finish()
        if any(box.isVisible() for box in window._advanced_boxes):
            fail("advanced settings are visible in the basic view")
        window.advanced_action.setChecked(True)
        if not all(box.isVisible() for box in window._advanced_boxes):
            fail("advanced settings did not appear")
        shot("ui_0_advanced.png")
        window.advanced_action.setChecked(False)
        QTimer.singleShot(2500, check_preview)

    def preview_images():
        return sum(1 for t in window.preview.tiles.values() if t._pixmap is not None)

    def check_preview():
        live = window.live
        if live is None or not live.running:
            fail("live preview is not running after discovery")
        n = preview_images()
        print(f"live preview: {n}/{len(window.preview.tiles)} tiles show an image")
        if n == 0:
            fail("live preview shows no images")
        shot("ui_1_preview.png")
        window.start_recording()
        state["start_deadline"] = time.time() + 120
        QTimer.singleShot(250, wait_started)

    def wait_started():
        # Arming includes waiting for PTP, which takes a variable time.
        session = window.session
        if session is not None and not session.is_running() and                 time.time() < state["start_deadline"]:
            QTimer.singleShot(250, wait_started)
            return
        QTimer.singleShot(int(args.seconds * 1000), mid_record)

    def mid_record():
        if window.session is None:
            fail("recording did not start")
            return finish()
        status = window.session.status()
        print(f"recording {status['elapsed_s']:.1f}s, spread "
              f"{status['sync_report']['spread_ms_max']:.4f} ms max, "
              f"{sum(c['grabbed'] for c in status['cameras'].values())} frames")
        if sum(c["grabbed"] for c in status["cameras"].values()) == 0:
            fail("no frames were grabbed")
        shot("ui_2_recording.png")
        window.stop_recording()
        state["stop_deadline"] = time.time() + 30
        QTimer.singleShot(500, finish)

    def finish():
        # Stopping drains the encoder queues, which takes a while on a slow machine:
        # wait for it rather than for a fixed time.
        if window.session is not None and time.time() < state.get("stop_deadline", 0):
            QTimer.singleShot(250, finish)
            return
        shot("ui_3_stopped.png")
        if window.session is not None:
            fail("session was not cleared after stop")
        QTimer.singleShot(3000, after_stop)

    def after_stop():
        live = window.live
        if live is None or not live.running:
            fail("live preview did not resume after the recording")
        shot("ui_4_preview_again.png")
        window.close()
        app.quit()

    QTimer.singleShot(2500, after_discover)
    QTimer.singleShot(int((args.seconds + 200) * 1000), app.quit)
    app.exec()

    print(f"\nscreenshots in {out}")
    if state["failures"]:
        print(f"UI SMOKE TEST FAILED: {len(state['failures'])} problem(s)")
        return 1
    print("UI SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
