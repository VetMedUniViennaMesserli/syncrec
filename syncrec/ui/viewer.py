"""Recording viewer: play all videos of a recording side by side, frame by frame.

    python -m syncrec.ui.viewer                      (then File > Open recording)
    python -m syncrec.ui.viewer recordings/20261012_141500

All videos always show the same video frame number. With the recorder's gap filling
(ptp_action) frame n is trigger n in every file, so what you see side by side was
exposed at the same moment. Each tile shows what the recorder's CSV says about the
frame: the trigger number, the camera's exposure timestamp relative to the median
of all cameras, and a red frame where the image is a placeholder (a missed frame
filled with the previous image).

Keys
    Space               play / pause
    Left / Right        one frame back / forward
    Shift + Left/Right  10 frames
    Ctrl + Left/Right   100 frames
    Home / End          first / last frame
    Mouse wheel         zoom the tile under the cursor (drag to pan)
    Double-click        show one camera large / back to the grid
    0                   reset zoom
    PageUp / PageDown   previous / next segment (split recordings)

Split recordings (segment_0000/, segment_0001/, ...): open the recording folder or
any segment. Stepping and playing run across segment boundaries; the status bar
shows the frame and time within the whole recording.

"Real frames only" (switched on automatically when most frames are placeholders,
e.g. because the cameras could only keep up with every 3rd trigger): stepping and
playing skip the placeholder frames and play at the rate the cameras really
recorded, in real time. A camera that has no new image at a shown frame (it took a
different trigger than the others) gets a small note instead of the red warning.
"""
from __future__ import annotations

import bisect
import json
import math
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import (QAction, QColor, QFont, QImage, QKeySequence, QPainter,
                           QPen, QShortcut)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog,
                               QGridLayout, QHBoxLayout, QLabel, QMainWindow,
                               QMessageBox, QPushButton, QSlider, QSpinBox,
                               QStatusBar, QVBoxLayout, QWidget)

from .video_reader import FrameInfo, VideoReader

VIDEO_EXT = (".mp4", ".mkv", ".avi", ".mov")


# ---- one camera tile ----------------------------------------------------------
class VideoTile(QWidget):
    doubleClicked = Signal(object)
    zoomChanged = Signal(object)

    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self.name = name
        self.image: Optional[QImage] = None
        self._array: Optional[np.ndarray] = None    # keeps QImage's buffer alive
        self.caption = ""
        self.placeholder = False
        self.note = ""
        self.missing = False
        self.zoom = 1.0
        self.center = QPointF(0.5, 0.5)            # view centre, image fractions
        self._drag: Optional[QPointF] = None
        self.setMinimumSize(160, 120)
        self.setMouseTracking(True)

    def set_frame(self, arr: Optional[np.ndarray], caption: str,
                  placeholder: bool, note: str = "") -> None:
        self.missing = arr is None
        if arr is not None:
            self._array = np.ascontiguousarray(arr)
            h, w, _ = self._array.shape
            self.image = QImage(self._array.data, w, h, 3 * w, QImage.Format_RGB888)
        self.caption = caption
        self.placeholder = placeholder
        self.note = note
        self.update()

    def set_view(self, zoom: float, center: QPointF) -> None:
        self.zoom, self.center = zoom, QPointF(center)
        self.update()

    def reset_zoom(self) -> None:
        self.set_view(1.0, QPointF(0.5, 0.5))

    # Geometry: where the image sits in the widget at the current zoom.
    def _target(self) -> QRectF:
        if self.image is None:
            return QRectF()
        iw, ih = self.image.width(), self.image.height()
        fit = min(self.width() / iw, self.height() / ih) * self.zoom
        w, h = iw * fit, ih * fit
        x = self.width() / 2 - self.center.x() * w
        y = self.height() / 2 - self.center.y() * h
        return QRectF(x, y, w, h)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(20, 20, 20))
        if self.image is not None and not self.missing:
            p.setRenderHint(QPainter.SmoothPixmapTransform, self.zoom < 2.0)
            p.drawImage(self._target(), self.image)
        if self.missing:
            p.setPen(QColor(200, 200, 200))
            p.drawText(self.rect(), Qt.AlignCenter, "no frame (video ended)")
        # Caption box.
        p.setFont(QFont("monospace", 9))
        lines = [self.name] + ([self.caption] if self.caption else [])
        if self.placeholder:
            lines.append("PLACEHOLDER - missed frame, previous image repeated")
        elif self.note:
            lines.append(self.note)
        fm = p.fontMetrics()
        bw = max(fm.horizontalAdvance(t) for t in lines) + 10
        bh = fm.height() * len(lines) + 6
        p.fillRect(QRectF(4, 4, bw, bh), QColor(0, 0, 0, 170))
        for i, text in enumerate(lines):
            last = i == len(lines) - 1
            p.setPen(QColor(255, 90, 90) if (self.placeholder and last)
                     else QColor(255, 190, 80) if (self.note and last)
                     else QColor(255, 255, 255))
            p.drawText(QPointF(9, 4 + fm.ascent() + 3 + i * fm.height()), text)
        if self.zoom > 1.0:
            p.setPen(QColor(255, 220, 0))
            p.drawText(QPointF(9, self.height() - 8), f"zoom {self.zoom:.1f}x")
        if self.placeholder:
            p.setPen(QPen(QColor(230, 40, 40), 4))
            p.drawRect(self.rect().adjusted(2, 2, -2, -2))
        p.end()

    def wheelEvent(self, event) -> None:
        if self.image is None:
            return
        t = self._target()
        pos = event.position()
        # Keep the image point under the cursor fixed while zooming.
        fx = (pos.x() - t.x()) / t.width()
        fy = (pos.y() - t.y()) / t.height()
        factor = 1.25 if event.angleDelta().y() > 0 else 1 / 1.25
        new_zoom = min(40.0, max(1.0, self.zoom * factor))
        scale = new_zoom / self.zoom
        self.zoom = new_zoom
        nw, nh = t.width() * scale, t.height() * scale
        self.center = QPointF(fx - (pos.x() - self.width() / 2) / nw,
                              fy - (pos.y() - self.height() / 2) / nh)
        if self.zoom == 1.0:
            self.center = QPointF(0.5, 0.5)
        self.update()
        self.zoomChanged.emit(self)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._drag = event.position()

    def mouseMoveEvent(self, event) -> None:
        if self._drag is None or self.image is None or self.zoom <= 1.0:
            return
        t = self._target()
        d = event.position() - self._drag
        self._drag = event.position()
        self.center = QPointF(self.center.x() - d.x() / t.width(),
                              self.center.y() - d.y() / t.height())
        self.update()
        self.zoomChanged.emit(self)

    def mouseReleaseEvent(self, _event) -> None:
        self._drag = None

    def mouseDoubleClickEvent(self, _event) -> None:
        self.doubleClicked.emit(self)


# ---- background loading -------------------------------------------------------
class FrameLoader(QObject):
    """Decodes the requested frame of every video in parallel, off the UI thread."""
    loaded = Signal(int, object)                   # index, {name: array or None}

    def __init__(self, readers: List[VideoReader]):
        super().__init__()
        self.readers = readers
        self.pool = ThreadPoolExecutor(max_workers=max(1, len(readers)))

    @Slot(int)
    def load(self, index: int) -> None:
        futures = {r.name: self.pool.submit(r.frame, index) for r in self.readers}
        frames = {}
        for name, fut in futures.items():
            try:
                frames[name] = fut.result()
            except Exception:
                frames[name] = None
        self.loaded.emit(index, frames)

    def shutdown(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)


# ---- main window ----------------------------------------------------------------
class ViewerWindow(QMainWindow):
    requestFrame = Signal(int)

    def __init__(self, folder: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("Recording viewer")
        self.resize(1400, 900)
        self.readers: List[VideoReader] = []
        self.tiles: List[VideoTile] = []
        self.index = 0
        self.total = 0
        self.fps = 25.0
        self.folder = ""
        self._busy = False
        self._pending: Optional[int] = None
        self._focused: Optional[VideoTile] = None
        self.real_slots: List[int] = []            # frames where any camera is real
        self.segment_root: Optional[str] = None
        self.segment_dirs: List[str] = []
        self.segment_index = 0
        self.base_frame = 0
        self._thread: Optional[QThread] = None
        self._loader: Optional[FrameLoader] = None
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._play_tick)

        self._build_ui()
        if folder:
            self.open_folder(folder)

    # ---- layout ----
    def _build_ui(self) -> None:
        central = QWidget()
        v = QVBoxLayout(central)
        v.setContentsMargins(6, 6, 6, 6)

        bar = QHBoxLayout()
        self.btn_open = QPushButton("Open recording...")
        self.btn_open.clicked.connect(self._choose_folder)
        bar.addWidget(self.btn_open)
        self.segment_combo = QComboBox()
        self.segment_combo.setToolTip("Segment of a split recording (PageUp / PageDown)")
        self.segment_combo.setFocusPolicy(Qt.NoFocus)
        self.segment_combo.currentIndexChanged.connect(self._segment_chosen)
        self.segment_combo.setVisible(False)
        bar.addWidget(self.segment_combo)
        bar.addSpacing(12)

        def button(text, tip, slot):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.setFocusPolicy(Qt.NoFocus)
            b.clicked.connect(slot)
            b.setFixedWidth(46)
            bar.addWidget(b)
            return b

        button("|<", "First frame (Home)", lambda: self.goto(0))
        button("-100", "100 back (Ctrl+Left)", lambda: self.step(-100))
        button("-10", "10 back (Shift+Left)", lambda: self.step(-10))
        button("-1", "1 back (Left)", lambda: self.step(-1))
        self.btn_play = button("Play", "Play / pause (Space)", self.toggle_play)
        self.btn_play.setFixedWidth(60)
        button("+1", "1 forward (Right)", lambda: self.step(1))
        button("+10", "10 forward (Shift+Right)", lambda: self.step(10))
        button("+100", "100 forward (Ctrl+Right)", lambda: self.step(100))
        button(">|", "Last frame (End)", lambda: self.goto(self.total - 1))
        bar.addSpacing(12)

        bar.addWidget(QLabel("Frame"))
        self.spin = QSpinBox()
        self.spin.setRange(0, 0)
        self.spin.setKeyboardTracking(False)
        self.spin.valueChanged.connect(self.goto)
        bar.addWidget(self.spin)
        self.lbl_total = QLabel("/ 0")
        bar.addWidget(self.lbl_total)
        bar.addSpacing(12)

        bar.addWidget(QLabel("Speed"))
        self.speed = QComboBox()
        for s in ("0.1x", "0.25x", "0.5x", "1x", "2x"):
            self.speed.addItem(s, float(s[:-1]))
        self.speed.setCurrentText("1x")
        self.speed.currentIndexChanged.connect(self._update_timer)
        self.speed.setFocusPolicy(Qt.NoFocus)
        bar.addWidget(self.speed)

        self.real_only = QCheckBox("Real frames only")
        self.real_only.setToolTip("Skip placeholder frames when stepping and playing; "
                                  "play at the rate the cameras actually recorded")
        self.real_only.setFocusPolicy(Qt.NoFocus)
        self.real_only.toggled.connect(lambda _on: self._refresh())
        bar.addWidget(self.real_only)

        self.link_zoom = QCheckBox("Link zoom")
        self.link_zoom.setToolTip("Zoom and pan all cameras together")
        self.link_zoom.setFocusPolicy(Qt.NoFocus)
        bar.addWidget(self.link_zoom)

        self.btn_save = QPushButton("Save frame")
        self.btn_save.setToolTip("Save the current frame of every camera as PNG")
        self.btn_save.setFocusPolicy(Qt.NoFocus)
        self.btn_save.clicked.connect(self.save_frames)
        bar.addWidget(self.btn_save)
        bar.addStretch(1)
        v.addLayout(bar)

        self.grid_host = QWidget()
        self.grid = QGridLayout(self.grid_host)
        self.grid.setSpacing(4)
        self.grid.setContentsMargins(0, 0, 0, 0)
        v.addWidget(self.grid_host, 1)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.setFocusPolicy(Qt.NoFocus)
        self.slider.valueChanged.connect(self.goto)
        v.addWidget(self.slider)

        self.setCentralWidget(central)
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.lbl_time = QLabel()
        self.status.addPermanentWidget(self.lbl_time)

        open_act = QAction("Open recording...", self)
        open_act.setShortcut(QKeySequence.Open)
        open_act.triggered.connect(self._choose_folder)
        self.addAction(open_act)
        self._install_shortcuts()

    # ---- opening ----
    def _choose_folder(self) -> None:
        start = self.folder or os.path.abspath("recordings")
        folder = QFileDialog.getExistingDirectory(self, "Open recording folder", start)
        if folder:
            self.open_folder(folder)

    def _ffmpeg_for(self, folder: str) -> str:
        """The ffmpeg the recording was made with (from its manifest), else PATH."""
        for where in (folder, os.path.dirname(os.path.abspath(folder))):
            try:
                with open(os.path.join(where, "manifest.json"), encoding="utf-8") as fh:
                    cand = json.load(fh).get("config", {}).get("encoder", {}).get("ffmpeg")
                if cand and (os.path.exists(cand) or shutil.which(cand)):
                    return cand
            except (OSError, ValueError):
                pass
        for cand in ("ffmpeg", r"C:\ffmpeg\bin\ffmpeg.exe",
                     r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"):
            if shutil.which(cand) or os.path.exists(cand):
                return cand
        return "ffmpeg"

    @staticmethod
    def _segments_of(folder: str):
        """(recording root, segment folders) for a split recording, else (None, [])."""
        def segs(root):
            try:
                return [os.path.join(root, d) for d in sorted(os.listdir(root))
                        if d.startswith("segment_") and os.path.isdir(os.path.join(root, d))]
            except OSError:
                return []
        folder = os.path.abspath(folder)
        found = segs(folder)
        if found:
            return folder, found
        if os.path.basename(folder).startswith("segment_"):
            parent = os.path.dirname(folder)
            return parent, segs(parent)
        return None, []

    def _segment_chosen(self, i: int) -> None:
        if 0 <= i < len(self.segment_dirs) and \
                os.path.abspath(self.folder) != self.segment_dirs[i]:
            self.open_folder(self.segment_dirs[i])

    def _change_segment(self, delta: int, start: int = 0) -> bool:
        """Open the neighbouring segment at frame `start` (-1 = its last frame)."""
        i = self.segment_index + delta
        if not self.segment_dirs or not 0 <= i < len(self.segment_dirs):
            return False
        playing = self._play_timer.isActive()
        self.open_folder(self.segment_dirs[i], start=start)
        if playing:
            self.toggle_play()
        return True

    def open_folder(self, folder: str, start: int = 0) -> None:
        root, segments = self._segments_of(folder)
        if segments and os.path.abspath(folder) == root:
            folder = segments[0]
        self.segment_root, self.segment_dirs = root, segments
        self.segment_index = (segments.index(os.path.abspath(folder))
                              if os.path.abspath(folder) in segments else 0)
        self.segment_combo.blockSignals(True)
        self.segment_combo.clear()
        for d in segments:
            self.segment_combo.addItem(os.path.basename(d))
        self.segment_combo.setCurrentIndex(self.segment_index)
        self.segment_combo.setVisible(bool(segments))
        self.segment_combo.blockSignals(False)
        videos = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                        if f.lower().endswith(VIDEO_EXT))
        if not videos:
            QMessageBox.warning(self, "No videos", f"No video files in\n{folder}")
            return
        self._close_videos()
        ffmpeg = self._ffmpeg_for(folder)
        readers = []
        for path in videos:
            try:
                readers.append(VideoReader(path, ffmpeg))
            except Exception as exc:
                QMessageBox.warning(self, "Cannot open video", f"{path}\n\n{exc}")
        if not readers:
            return
        self.folder = folder
        self.readers = readers
        self.fps = readers[0].fps or 25.0
        self.total = max(r.frame_count for r in readers)
        title = os.path.basename(root) + "/" + os.path.basename(folder) if segments \
            else os.path.basename(folder)
        self.setWindowTitle(f"Recording viewer - {title}")
        # Frame number of this file's first frame within the whole recording.
        first = readers[0].frame_info(0)
        self.base_frame = first.trigger_id if first is not None and first.trigger_id > 0 else 0

        for i, r in enumerate(readers):
            tile = VideoTile(r.name)
            tile.doubleClicked.connect(self._toggle_focus)
            tile.zoomChanged.connect(self._zoom_changed)
            self.tiles.append(tile)
        self._layout_tiles()

        self._thread = QThread(self)
        self._loader = FrameLoader(self.readers)
        self._loader.moveToThread(self._thread)
        self.requestFrame.connect(self._loader.load)
        self._loader.loaded.connect(self._show_frames)
        self._thread.start()

        for w in (self.spin, self.slider):
            w.blockSignals(True)
            w.setRange(0, max(0, self.total - 1))
            w.blockSignals(False)
        self.lbl_total.setText(f"/ {max(0, self.total - 1)}")
        self._find_real_frames()
        lengths = {r.name: r.frame_count for r in readers}
        note = ""
        if len(set(lengths.values())) > 1:
            note = "  |  videos differ in length: " + ", ".join(
                f"{n} {c}" for n, c in lengths.items())
        real_note = ""
        if len(self.real_slots) < self.total:
            rates = []
            common = None
            for r in readers:
                n = r.frame_count or len(r.info)
                real = {i for i in range(n)
                        if not (r.frame_info(i) and r.frame_info(i).placeholder)}
                rates.append(self.fps * len(real) / max(1, n))
                common = real if common is None else common & real
            real_note = f"  |  cameras recorded about {sum(rates) / len(rates):.1f} fps"
            if common is not None and len(common) < 0.9 * len(self.real_slots):
                real_note += (" - NOT all at the same triggers (frames without a new "
                              "image are noted per camera)")
        self.status.showMessage(f"{len(readers)} videos, {self.total} frames at "
                                f"{self.fps:g} fps{real_note}{note}")
        self.index = -1
        self.goto(self.total - 1 if start < 0 else start)

    def _layout_tiles(self) -> None:
        while self.grid.count():
            self.grid.takeAt(0)
        shown = [self._focused] if self._focused else self.tiles
        for t in self.tiles:
            t.setVisible(t in shown)
        cols = max(1, math.ceil(math.sqrt(len(shown) * 4 / 3)))
        for i, t in enumerate(shown):
            self.grid.addWidget(t, i // cols, i % cols)

    def _close_videos(self) -> None:
        self._play_timer.stop()
        self.btn_play.setText("Play")
        if self._loader is not None:
            self.requestFrame.disconnect(self._loader.load)
            self._loader.shutdown()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(3000)
        for r in self.readers:
            r.close()
        for t in self.tiles:
            t.setParent(None)
            t.deleteLater()
        self.readers, self.tiles = [], []
        self._focused = None
        self._loader = self._thread = None
        self._busy, self._pending = False, None

    # ---- navigation ----
    def goto(self, index: int) -> None:
        if not self.readers:
            return
        index = max(0, min(int(index), self.total - 1))
        if index == self.index and not self._busy:
            return
        self.index = index
        for w in (self.spin, self.slider):
            if w.value() != index:
                w.blockSignals(True)
                w.setValue(index)
                w.blockSignals(False)
        if self._busy:
            self._pending = index                  # only the latest request matters
        else:
            self._busy = True
            self.requestFrame.emit(index)

    def _find_real_frames(self) -> None:
        """Frames where at least one camera has a real image; decide the mode."""
        slots = set()
        placeholders = 0
        for r in self.readers:
            n = r.frame_count or len(r.info)
            for i in range(n):
                info = r.frame_info(i)
                if info is not None and info.placeholder:
                    placeholders += 1
                else:
                    slots.add(i)
        self.real_slots = sorted(s for s in slots if s < self.total)
        frames = sum((r.frame_count or len(r.info)) for r in self.readers)
        # Mostly placeholders: the cameras ran slower than the trigger rate.
        self.real_only.blockSignals(True)
        self.real_only.setChecked(frames > 0 and placeholders / frames > 0.3)
        self.real_only.blockSignals(False)

    def _real_mode(self) -> bool:
        return self.real_only.isChecked() and bool(self.real_slots)

    def _neighbour(self, index: int, delta: int) -> int:
        """`delta` real frames away from `index` (index itself need not be real)."""
        if not self._real_mode() or delta == 0:
            return index + delta
        s = self.real_slots
        if delta > 0:
            pos = bisect.bisect_right(s, index) + delta - 1
        else:
            pos = bisect.bisect_left(s, index) + delta
        return s[max(0, min(len(s) - 1, pos))]

    def step(self, delta: int) -> None:
        if self._play_timer.isActive():
            self.toggle_play()
        target = self._neighbour(self.index, delta)
        # Past either end of a segment: continue in the neighbouring one.
        if delta > 0 and self.index >= self.total - 1 and self._change_segment(1, 0):
            return
        if delta < 0 and self.index <= 0 and self._change_segment(-1, -1):
            return
        self.goto(target)

    def _jump(self, index: int) -> None:
        if self._play_timer.isActive():
            self.toggle_play()
        self.goto(index)

    def _refresh(self) -> None:
        if self.readers:
            i, self.index = self.index, -1
            self.goto(i)

    def toggle_play(self) -> None:
        if not self.readers:
            return
        if self._play_timer.isActive():
            self._play_timer.stop()
            self.btn_play.setText("Play")
        else:
            if self._neighbour(self.index, 1) <= self.index or self.index >= self.total - 1:
                self.goto(self._neighbour(-1, 1))
            self._update_timer()
            self._play_timer.start()
            self.btn_play.setText("Pause")

    def _update_timer(self, frames: int = 1) -> None:
        # One tick shows the next frame; in real-frames mode that can be several
        # trigger periods ahead, so the tick lasts that long and playback stays real time.
        speed = self.speed.currentData() or 1.0
        self._play_timer.setInterval(max(1, int(1000 * max(1, frames) / (self.fps * speed))))

    def _play_tick(self) -> None:
        if self._busy:
            return                                 # decoding is the limit; never skip
        nxt = self._neighbour(self.index, 1)
        if nxt <= self.index or nxt > self.total - 1:
            if not self._change_segment(1, 0):           # keeps playing if it exists
                self.toggle_play()
            return
        self.goto(nxt)
        after = self._neighbour(nxt, 1)
        self._update_timer(after - nxt if after > nxt else 1)

    @Slot(int, object)
    def _show_frames(self, index: int, frames: Dict[str, Optional[np.ndarray]]) -> None:
        infos = {r.name: r.frame_info(index) for r in self.readers}
        stamps = [i.cam_ns for i in infos.values()
                  if i is not None and i.cam_ns and not i.placeholder]
        ref = float(np.median(stamps)) if len(stamps) > 1 else None
        placeholders = []
        real_mode = self._real_mode()
        for tile in self.tiles:
            info: Optional[FrameInfo] = infos.get(tile.name)
            parts = [f"frame {index}"]
            if info is not None:
                if info.trigger_id >= 0:
                    parts.append(f"trigger {info.trigger_id}")
                if ref is not None and info.cam_ns and not info.placeholder:
                    parts.append(f"dt {(info.cam_ns - ref) / 1000:+.1f} us")
                if info.placeholder:
                    placeholders.append(tile.name)
            is_ph = bool(info and info.placeholder)
            note = ""
            if is_ph and real_mode:
                # Expected in this mode: say which trigger the image really is from.
                reader = next(r for r in self.readers if r.name == tile.name)
                k = index
                while k > 0 and (reader.frame_info(k) is None or reader.frame_info(k).placeholder):
                    k -= 1
                src = reader.frame_info(k)
                age = index - k
                note = (f"no new image here - showing frame {k}"
                        + (f" (trigger {src.trigger_id})" if src and src.trigger_id >= 0 else "")
                        + f", {age} earlier")
            tile.set_frame(frames.get(tile.name), "  ".join(parts),
                           is_ph and not real_mode, note)
        absolute = getattr(self, "base_frame", 0) + index
        t = absolute / self.fps
        where = (f"segment {self.segment_index}  frame {index}  |  recording frame "
                 f"{absolute}" if self.segment_dirs else f"frame {index}")
        self.lbl_time.setText(f"{int(t // 3600):d}:{int(t % 3600 // 60):02d}:"
                              f"{t % 60:06.3f}   {where}"
                              + (f"   no new image: {', '.join(placeholders)}"
                                 if placeholders and real_mode else
                                 f"   placeholder: {', '.join(placeholders)}"
                                 if placeholders else ""))
        self._busy = False
        if self._pending is not None and self._pending != index:
            nxt, self._pending = self._pending, None
            self._busy = True
            self.requestFrame.emit(nxt)
        else:
            self._pending = None

    # ---- zoom / focus ----
    def _zoom_changed(self, tile: VideoTile) -> None:
        if self.link_zoom.isChecked():
            for t in self.tiles:
                if t is not tile:
                    t.set_view(tile.zoom, tile.center)

    def _toggle_focus(self, tile: VideoTile) -> None:
        self._focused = None if self._focused is tile else tile
        self._layout_tiles()

    def save_frames(self) -> None:
        if not self.readers:
            return
        out = os.path.join(self.folder, "frames", f"frame_{self.index:06d}")
        os.makedirs(out, exist_ok=True)
        saved = 0
        for t in self.tiles:
            if t.image is not None and not t.missing:
                t.image.save(os.path.join(out, f"{t.name}.png"))
                saved += 1
        self.status.showMessage(f"saved {saved} images to {out}", 5000)

    # ---- keys ----
    def _install_shortcuts(self) -> None:
        # Window-wide shortcuts, so they work whichever control has focus.
        keys = [("Right", lambda: self.step(1)), ("Left", lambda: self.step(-1)),
                ("Shift+Right", lambda: self.step(10)), ("Shift+Left", lambda: self.step(-10)),
                ("Ctrl+Right", lambda: self.step(100)), ("Ctrl+Left", lambda: self.step(-100)),
                ("Space", self.toggle_play), ("Home", lambda: self._jump(0)),
                ("End", lambda: self._jump(self.total - 1)), ("0", self._reset_zoom),
                ("PgUp", lambda: self._change_segment(-1, 0)),
                ("PgDown", lambda: self._change_segment(1, 0))]
        for seq, fn in keys:
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.WindowShortcut)
            sc.activated.connect(fn)

    def _reset_zoom(self) -> None:
        for t in self.tiles:
            t.reset_zoom()

    def closeEvent(self, event) -> None:
        self._close_videos()
        super().closeEvent(event)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    app = QApplication.instance() or QApplication(sys.argv[:1])
    win = ViewerWindow(argv[0] if argv else None)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
