"""The recorder's Qt desktop UI.

Arming a recording can block for a long time -- PTP convergence alone is allowed up
to a minute -- so discovery, the live preview, prepare and stop all run on worker
threads and report back by signal. Only one worker runs at a time. The GUI thread
only ever polls the preview or the session for numbers and thumbnails, which keeps
the window responsive while many cameras stream.

The window opens in a basic view with the everyday settings (which cameras,
where, what name, frame rate, exposure, gain). Settings > Show advanced settings
reveals everything else, including saving the settings as the new default.
"""
from __future__ import annotations

import copy
import logging
import os
import shutil
import sys
import traceback
from datetime import datetime
from typing import List, Optional

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QFont, QKeySequence, QPalette
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QMainWindow,
                               QMessageBox, QPlainTextEdit, QPushButton, QScrollArea,
                               QSizePolicy, QSpinBox, QSplitter, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from ..backends import discover
from ..camera import CameraBackend
from ..config import DEFAULT_CONFIG_PATH, CameraOverride, RecorderConfig
from ..encoders import CODECS, GPU_ORDER, HARDWARE, build_args, \
    container_of, encoder_name, test as test_encoder
from ..live import LivePreview
from ..session import (CLIP_VARIABLES, DEFAULT_CLIP_NAME, RecordingSession,
                       check_clip_name, expand_clip_name)
from .preview import PreviewGrid

log = logging.getLogger(__name__)

SYNC_LABELS = {
    "auto": "Auto (best the cameras support)",
    "ptp_action": "PTP + scheduled action commands (GigE)",
    "hardware_trigger": "Hardware trigger line",
    "software_trigger": "Software trigger (host paced)",
    "freerun": "Free run (NOT synchronised)",
}

ADVANCED_SUFFIX = "  (advanced)"


class QtLogHandler(logging.Handler, QObject):
    """Feeds the logging module into the UI's log pane."""
    message = Signal(str)

    def __init__(self):
        logging.Handler.__init__(self)
        QObject.__init__(self)
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s",
                                            datefmt="%H:%M:%S"))

    def emit(self, record) -> None:
        try:
            self.message.emit(self.format(record))
        except Exception:
            pass


class Worker(QThread):
    """Runs one blocking callable off the GUI thread.

    The signals carry the worker itself, so the window can find the callbacks that
    belong to it.
    """
    finished_ok = Signal(object, object)
    failed = Signal(object, str)

    def __init__(self, fn, on_ok, on_err):
        super().__init__()
        self._fn = fn
        self.on_ok = on_ok
        self.on_err = on_err

    def run(self) -> None:
        try:
            result = self._fn()
        except Exception as exc:
            log.debug("worker failed:\n%s", traceback.format_exc())
            self.failed.emit(self, str(exc))
        else:
            self.finished_ok.emit(self, result)


class MainWindow(QMainWindow):
    def __init__(self, config_path: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("Synchronized Multi-Camera Recorder")
        self.resize(1500, 950)

        # The settings file loaded at start; "Save as default" writes back to it.
        self.default_path = os.path.abspath(config_path or DEFAULT_CONFIG_PATH)
        self.cfg = RecorderConfig()
        load_error = ""
        if os.path.isfile(self.default_path):
            try:
                self.cfg = RecorderConfig.load(self.default_path)
            except Exception as exc:
                load_error = str(exc)
        self.cameras: List[CameraBackend] = []
        self.session: Optional[RecordingSession] = None
        self.live: Optional[LivePreview] = None
        self._workers: List[Worker] = []
        self._busy = False
        self._loading = False
        self._backend_name = ""

        # Preview restarts are debounced: typing a resolution should not restart
        # every camera per keystroke.
        self._restart_timer = QTimer(self)
        self._restart_timer.setSingleShot(True)
        self._restart_timer.setInterval(600)
        self._restart_timer.timeout.connect(self._start_preview)

        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self._poll)
        self.poll_timer.setInterval(200)

        self._build_ui()
        self._build_menu()
        self._install_log_handler()
        self._apply_config_to_widgets()
        self._set_advanced(False)
        self._update_buttons()

        if load_error:
            log.error("could not load %s: %s -- using built-in defaults",
                      self.default_path, load_error)
            QMessageBox.warning(self, "Settings not loaded",
                                f"{self.default_path} could not be read:\n\n{load_error}"
                                f"\n\nBuilt-in defaults are used instead.")
        elif os.path.isfile(self.default_path):
            log.info("loaded settings from %s", self.default_path)
        else:
            log.info("%s not found, using built-in defaults", self.default_path)

        self.poll_timer.start()
        QTimer.singleShot(150, self.discover_cameras)

    # ---- construction ------------------------------------------------------
    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Horizontal, self)
        splitter.addWidget(self._build_left_panel())

        right = QSplitter(Qt.Vertical, self)
        self.preview = PreviewGrid(self)
        right.addWidget(self.preview)
        right.addWidget(self._build_bottom_panel())
        right.setStretchFactor(0, 3)
        right.setStretchFactor(1, 1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([440, 1060])
        self.setCentralWidget(splitter)

        self.sync_label = QLabel("no cameras")
        self.disk_label = QLabel("")
        self.rate_label = QLabel("")
        bar = self.statusBar()
        bar.addWidget(self.sync_label, 1)
        bar.addPermanentWidget(self.rate_label)
        bar.addPermanentWidget(self.disk_label)

    def _build_menu(self) -> None:
        menu = self.menuBar().addMenu("&Settings")
        self.advanced_action = QAction("Show advanced settings", self)
        self.advanced_action.setCheckable(True)
        self.advanced_action.setShortcut(QKeySequence("Ctrl+Shift+A"))
        self.advanced_action.toggled.connect(self._set_advanced)
        menu.addAction(self.advanced_action)

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(8, 8, 8, 8)

        # -- transport buttons
        buttons = QHBoxLayout()
        self.btn_discover = QPushButton("Find cameras")
        self.btn_discover.setToolTip("Search for connected cameras again")
        self.btn_discover.clicked.connect(self.discover_cameras)
        self.btn_record = QPushButton("● Record")
        self.btn_record.setStyleSheet("QPushButton { font-weight: bold; padding: 8px; }")
        self.btn_record.clicked.connect(self.start_recording)
        self.btn_stop = QPushButton("■ Stop")
        self.btn_stop.setStyleSheet("QPushButton { font-weight: bold; padding: 8px; }")
        self.btn_stop.clicked.connect(self.stop_recording)
        buttons.addWidget(self.btn_discover)
        buttons.addWidget(self.btn_record, 1)
        buttons.addWidget(self.btn_stop, 1)
        outer.addLayout(buttons)

        self.elapsed_label = QLabel("00:00:00")
        font = QFont("Consolas")
        font.setPointSize(20)
        self.elapsed_label.setFont(font)
        self.elapsed_label.setAlignment(Qt.AlignCenter)
        outer.addWidget(self.elapsed_label)

        # Camera list and settings share a splitter, so the list can be dragged as
        # tall as it needs to be.
        self.left_split = QSplitter(Qt.Vertical)
        self.left_split.setChildrenCollapsible(False)
        self.left_split.addWidget(self._build_camera_group())

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        form_layout = QVBoxLayout(inner)
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.addWidget(self._build_recording_group())
        form_layout.addWidget(self._build_camera_settings_group())
        self.sync_box = self._build_sync_group()
        self.image_box = self._build_image_group()
        self.encoding_box = self._build_encoding_group()
        self._advanced_boxes = [self.sync_box, self.image_box, self.encoding_box]
        for box in self._advanced_boxes:
            box.setTitle(box.title() + ADVANCED_SUFFIX)
            form_layout.addWidget(box)
        form_layout.addStretch(1)
        scroll.setWidget(inner)
        self.settings_scroll = scroll
        self.left_split.addWidget(scroll)
        self.left_split.setStretchFactor(0, 0)
        self.left_split.setStretchFactor(1, 1)
        outer.addWidget(self.left_split, 1)

        self.config_buttons = QWidget()
        config_row = QHBoxLayout(self.config_buttons)
        config_row.setContentsMargins(0, 0, 0, 0)
        btn_load = QPushButton("Load config…")
        btn_load.clicked.connect(self.load_config)
        btn_save = QPushButton("Save config as…")
        btn_save.clicked.connect(self.save_config)
        btn_default = QPushButton("Save as default")
        btn_default.setToolTip(f"Write the current settings to {self.default_path}; "
                               f"they are loaded every time the program starts")
        btn_default.clicked.connect(self.save_default)
        config_row.addWidget(btn_load)
        config_row.addWidget(btn_save)
        config_row.addWidget(btn_default)
        outer.addWidget(self.config_buttons)
        return panel

    def _build_camera_group(self) -> QGroupBox:
        box = QGroupBox("Cameras")
        self.camera_box = box
        layout = QVBoxLayout(box)
        self.camera_table = QTableWidget(0, 3)
        self.camera_table.setHorizontalHeaderLabels(["Use", "Serial", "Model"])
        self.camera_table.verticalHeader().setVisible(False)
        self.camera_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch)
        self.camera_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents)
        self.camera_table.setMinimumHeight(80)
        self.camera_table.setToolTip("Untick a camera to leave it out of the recording")
        layout.addWidget(self.camera_table, 1)
        self.backend_label = QLabel("backend: -")
        self.backend_label.setStyleSheet("color: #8b949e;")
        layout.addWidget(self.backend_label)
        return box

    def _build_recording_group(self) -> QGroupBox:
        box = QGroupBox("Recording")
        form = QFormLayout(box)
        dir_row = QHBoxLayout()
        self.output_dir = QLineEdit()
        self.output_dir.setToolTip("Every recording gets its own folder inside this one")
        btn_browse = QPushButton("…")
        btn_browse.setMaximumWidth(32)
        btn_browse.clicked.connect(self._browse_output)
        dir_row.addWidget(self.output_dir)
        dir_row.addWidget(btn_browse)
        dir_widget = QWidget()
        dir_widget.setLayout(dir_row)
        dir_row.setContentsMargins(0, 0, 0, 0)
        form.addRow("Folder", dir_widget)

        name_row = QHBoxLayout()
        self.clip_name = QLineEdit()
        self.clip_name.setPlaceholderText(DEFAULT_CLIP_NAME)
        self.clip_name.setToolTip(
            "Name of the recording's folder and files. {date} and {time} are filled "
            "in when you press Record. Press ? for examples.")
        btn_help = QPushButton("?")
        btn_help.setMaximumWidth(32)
        btn_help.setToolTip("How to name a recording")
        btn_help.clicked.connect(self._show_name_help)
        name_row.addWidget(self.clip_name)
        name_row.addWidget(btn_help)
        name_widget = QWidget()
        name_widget.setLayout(name_row)
        name_row.setContentsMargins(0, 0, 0, 0)
        form.addRow("Recording name", name_widget)

        self.name_example = QLabel("")
        self.name_example.setWordWrap(True)
        # Long paths must wrap, not widen the panel.
        self.name_example.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.name_example.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form.addRow("", self.name_example)
        for edit in (self.output_dir, self.clip_name):
            edit.textChanged.connect(self._update_name_example)
        return box

    def _build_camera_settings_group(self) -> QGroupBox:
        box = QGroupBox("Camera settings")
        form = QFormLayout(box)
        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(0.1, 1000.0)
        self.fps_spin.setSuffix(" fps")
        self.fps_spin.setToolTip(
            "Images per second. Higher rates need more network bandwidth and disk "
            "space; if the camera tiles report dropped or missed frames, lower it.")
        form.addRow("Frame rate", self.fps_spin)

        self.exposure_spin = QDoubleSpinBox()
        self.exposure_spin.setRange(1.0, 1e7)
        self.exposure_spin.setDecimals(0)
        self.exposure_spin.setSuffix(" µs")
        self.exposure_spin.setSingleStep(100)
        self.exposure_spin.setToolTip(
            "How long each image is exposed (1000 µs = 1 ms). Longer is brighter but "
            "blurs movement, and it must be shorter than one frame (1 / frame rate). "
            "The live preview shows the effect immediately.")
        form.addRow("Exposure", self.exposure_spin)

        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setRange(0.0, 48.0)
        self.gain_spin.setSuffix(" dB")
        self.gain_spin.setToolTip(
            "Electronic brightening. Makes the image brighter but also noisier; "
            "prefer more light or a longer exposure where possible.")
        form.addRow("Gain", self.gain_spin)

        for spin in (self.fps_spin, self.exposure_spin, self.gain_spin):
            spin.valueChanged.connect(self._live_setting_changed)
        return box

    def _build_sync_group(self) -> QGroupBox:
        box = QGroupBox("Synchronisation")
        form = QFormLayout(box)
        self.sync_combo = QComboBox()
        for key, label in SYNC_LABELS.items():
            self.sync_combo.addItem(label, key)
        self.sync_combo.currentIndexChanged.connect(self._sync_mode_changed)
        form.addRow("Mode", self.sync_combo)

        self.trigger_source = QComboBox()
        self.trigger_source.setEditable(True)
        self.trigger_source.addItems(["Line1", "Line2", "Line3", "Line4"])
        form.addRow("Trigger line", self.trigger_source)

        self.trigger_activation = QComboBox()
        self.trigger_activation.addItems(["RisingEdge", "FallingEdge", "AnyEdge",
                                          "LevelHigh", "LevelLow"])
        form.addRow("Activation", self.trigger_activation)

        self.master_serial = QComboBox()
        self.master_serial.setEditable(True)
        self.master_serial.addItem("(external signal generator)")
        form.addRow("Trigger master", self.master_serial)

        self.max_spread = QDoubleSpinBox()
        self.max_spread.setRange(0.001, 1000.0)
        self.max_spread.setDecimals(3)
        self.max_spread.setSuffix(" ms")
        form.addRow("Warn above", self.max_spread)
        return box

    def _build_image_group(self) -> QGroupBox:
        box = QGroupBox("Image format")
        form = QFormLayout(box)
        self.backend_combo = QComboBox()
        self.backend_combo.addItem("Auto (Basler, else dummy)", "auto")
        self.backend_combo.addItem("Basler / pylon", "pylon")
        self.backend_combo.addItem("Dummy cameras", "dummy")
        self.backend_combo.setToolTip("Takes effect when you press Find cameras")
        form.addRow("Backend", self.backend_combo)

        size_row = QHBoxLayout()
        self.width_spin = QSpinBox()
        self.width_spin.setRange(16, 20000)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(16, 20000)
        size_row.addWidget(self.width_spin)
        size_row.addWidget(QLabel("×"))
        size_row.addWidget(self.height_spin)
        size_widget = QWidget()
        size_widget.setLayout(size_row)
        size_row.setContentsMargins(0, 0, 0, 0)
        form.addRow("Resolution", size_widget)

        self.binning_combo = QComboBox()
        for b, text in ((1, "off (crop only)"), (2, "2 x 2"), (3, "3 x 3"), (4, "4 x 4")):
            self.binning_combo.addItem(text, b)
        self.binning_combo.setToolTip(
            "Make the image smaller WITHOUT cropping: the camera combines pixels, so the "
            "whole field of view arrives with less data. Resolution is then in binned "
            "pixels (2048x1536 sensor, 2 x 2 -> 1024x768 for the full view). Colour "
            "models usually cannot bin.")
        form.addRow("Binning", self.binning_combo)

        self.pixfmt_combo = QComboBox()
        self.pixfmt_combo.setEditable(True)
        self.pixfmt_combo.addItems(["Mono8", "Mono12", "BayerRG8", "BayerBG8",
                                    "BayerGR8", "BayerGB8", "RGB8Packed"])
        form.addRow("Pixel format", self.pixfmt_combo)

        self.queue_spin = QSpinBox()
        self.queue_spin.setRange(2, 10000)
        self.queue_spin.setToolTip(
            "Frames buffered per camera before the recorder starts dropping them.\n"
            "Raise it if the encoder stutters; it costs RAM (frame size × cameras).")
        form.addRow("Queue depth", self.queue_spin)

        # These need the cameras reconfigured, i.e. the preview restarted.
        self.width_spin.valueChanged.connect(self._schedule_preview_restart)
        self.height_spin.valueChanged.connect(self._schedule_preview_restart)
        self.binning_combo.currentIndexChanged.connect(self._schedule_preview_restart)
        self.pixfmt_combo.currentTextChanged.connect(self._schedule_preview_restart)
        return box

    def _build_encoding_group(self) -> QGroupBox:
        box = QGroupBox("Files & encoding")
        form = QFormLayout(box)
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.0, 86400.0)
        self.duration_spin.setSuffix(" s")
        self.duration_spin.setSpecialValueText("until Stop")
        self.duration_spin.setToolTip("0 = record until you press Stop")
        form.addRow("Duration", self.duration_spin)

        self.pattern_edit = QLineEdit()
        self.pattern_edit.setToolTip("Placeholders: {clip} (the recording name), "
                                     "{serial}, {name}")
        self.pattern_edit.textChanged.connect(self._update_name_example)
        form.addRow("File name", self.pattern_edit)

        self.segment_spin = QDoubleSpinBox()
        self.segment_spin.setRange(0.0, 24 * 60.0)
        self.segment_spin.setDecimals(1)
        self.segment_spin.setSuffix(" min")
        self.segment_spin.setSpecialValueText("off (one file)")
        self.segment_spin.setToolTip(
            "Split the recording into segments of this length, each in its own "
            "folder (segment_0000, ...). Happens inside the running encoder at an "
            "exact frame: no frame is lost, all cameras split at the same trigger.")
        self.segment_spin.valueChanged.connect(self._update_name_example)
        form.addRow("Split every", self.segment_spin)

        self.codec_combo = QComboBox()
        for key, label in CODECS.items():
            self.codec_combo.addItem(label, key)
        self.codec_combo.addItem("Custom ffmpeg arguments", "custom")
        self.codec_combo.currentIndexChanged.connect(self._encoder_changed)
        form.addRow("Codec", self.codec_combo)

        self.hw_combo = QComboBox()
        for key, label in HARDWARE.items():
            self.hw_combo.addItem(label, key)
        self.hw_combo.setToolTip(
            "Where to encode. A GPU takes the load off the CPU, which matters with "
            "many cameras. Auto tries the GPUs and falls back to the CPU.")
        self.hw_combo.currentIndexChanged.connect(self._encoder_changed)
        self.test_encoder_btn = QPushButton("Test")
        self.test_encoder_btn.setToolTip(
            "Check which encoders work on this PC, with one encode per camera at once")
        self.test_encoder_btn.clicked.connect(self._test_encoders)
        hw_row = QWidget()
        hw_lay = QHBoxLayout(hw_row)
        hw_lay.setContentsMargins(0, 0, 0, 0)
        hw_lay.addWidget(self.hw_combo, 1)
        hw_lay.addWidget(self.test_encoder_btn)
        form.addRow("Encode on", hw_row)

        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(-1, 63)
        self.quality_spin.setSpecialValueText("codec default")
        self.quality_spin.setToolTip("Lower = better quality and bigger files. "
                                     "Defaults: H.264 20, H.265 24, AV1 32.")
        self.quality_spin.valueChanged.connect(self._encoder_changed)
        form.addRow("Quality", self.quality_spin)

        self.codec_args = QLineEdit()
        self.codec_args.setToolTip("ffmpeg output arguments. Generated from the "
                                   "settings above; editable with 'Custom'.")
        self.codec_args.textEdited.connect(self._custom_args_edited)
        form.addRow("ffmpeg args", self.codec_args)

        self.ffmpeg_path = QLineEdit()
        form.addRow("ffmpeg", self.ffmpeg_path)
        return box

    def _build_bottom_panel(self) -> QWidget:
        panel = QSplitter(Qt.Horizontal)
        self.stats_table = QTableWidget(0, 8)
        self.stats_table.setHorizontalHeaderLabels(
            ["Camera", "fps", "Grabbed", "Written", "Dropped", "Missed", "Filled",
             "Queue"])
        tips = {2: "Frames received from the camera",
                3: "Real frames written to the video",
                4: "Received, but discarded because encoding could not keep up (PC side)",
                5: "Lost between camera and PC: incomplete, or never arrived (network)",
                6: "Video slots filled with the previous image: missed, dropped, or "
                   "triggers the camera ignored (ptp_action only)",
                7: "Frames waiting to be encoded"}
        for col, tip in tips.items():
            self.stats_table.horizontalHeaderItem(col).setToolTip(tip)
        self.stats_table.verticalHeader().setVisible(False)
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        panel.addWidget(self.stats_table)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        self.log_view.setFont(QFont("Consolas", 9))
        panel.addWidget(self.log_view)
        panel.setSizes([700, 500])
        return panel

    def _install_log_handler(self) -> None:
        handler = QtLogHandler()
        handler.message.connect(self.log_view.appendPlainText)
        handler.setLevel(logging.INFO)
        logging.getLogger("syncrec").addHandler(handler)
        logging.getLogger("syncrec").setLevel(logging.INFO)
        root = logging.getLogger()
        if not root.handlers:
            logging.basicConfig(level=logging.INFO)

    # ---- basic / advanced view ---------------------------------------------
    def _set_advanced(self, on: bool) -> None:
        for box in self._advanced_boxes:
            box.setVisible(on)
        self.config_buttons.setVisible(on)
        if self.advanced_action.isChecked() != on:
            self.advanced_action.setChecked(on)

    # ---- config <-> widgets ------------------------------------------------
    def _apply_config_to_widgets(self) -> None:
        cfg = self.cfg
        self._loading = True          # no preview restarts for every widget we set
        try:
            self.backend_combo.setCurrentIndex(
                max(0, self.backend_combo.findData(cfg.acquisition.backend)))
            self.sync_combo.setCurrentIndex(
                max(0, self.sync_combo.findData(cfg.sync.mode)))
            self.trigger_source.setCurrentText(cfg.sync.hardware.trigger_source)
            self.trigger_activation.setCurrentText(cfg.sync.hardware.trigger_activation)
            if cfg.sync.hardware.master_serial:
                self.master_serial.setCurrentText(cfg.sync.hardware.master_serial)
            self.max_spread.setValue(cfg.sync.max_spread_ms)

            self.fps_spin.setValue(cfg.acquisition.fps)
            self.exposure_spin.setValue(cfg.acquisition.exposure_us)
            self.gain_spin.setValue(cfg.acquisition.gain_db)
            i = self.binning_combo.findData(
                int(getattr(cfg.acquisition, "binning", 1) or 1))
            self.binning_combo.setCurrentIndex(max(0, i))
            self.width_spin.setValue(cfg.acquisition.roi.width)
            self.height_spin.setValue(cfg.acquisition.roi.height)
            self.pixfmt_combo.setCurrentText(cfg.acquisition.pixel_format)
            self.queue_spin.setValue(cfg.acquisition.queue_size)

            self.output_dir.setText(cfg.session.output_dir)
            self.clip_name.setText(cfg.session.clip_name)
            self.pattern_edit.setText(cfg.session.filename_pattern)
            self.duration_spin.setValue(cfg.session.duration_s)
            self.segment_spin.setValue(cfg.session.segment_minutes)
            self.ffmpeg_path.setText(cfg.encoder.ffmpeg)
            self._custom_args = " ".join(cfg.encoder.output_args)
            codec = (cfg.encoder.codec or "custom").lower()
            self._set_combo(self.codec_combo, codec if codec in CODECS else "custom")
            self._set_combo(self.hw_combo, (cfg.encoder.hardware or "auto").lower())
            self.quality_spin.setValue(-1 if cfg.encoder.quality is None
                                       else int(cfg.encoder.quality))
            for row in range(self.camera_table.rowCount()):
                serial = self.camera_table.item(row, 1).text()
                self.camera_table.cellWidget(row, 0).setChecked(
                    cfg.override_for(serial).enabled)
        finally:
            self._loading = False
        self._encoder_changed()
        self._sync_mode_changed()
        self._update_name_example()

    def _collect_config(self) -> RecorderConfig:
        cfg = self.cfg
        cfg.acquisition.backend = self.backend_combo.currentData()
        cfg.sync.mode = self.sync_combo.currentData()
        cfg.sync.hardware.trigger_source = self.trigger_source.currentText().strip()
        cfg.sync.hardware.trigger_activation = self.trigger_activation.currentText()
        master = self.master_serial.currentText().strip()
        cfg.sync.hardware.master_serial = None if master.startswith("(") or not master \
            else master
        cfg.sync.max_spread_ms = self.max_spread.value()

        cfg.acquisition.fps = self.fps_spin.value()
        cfg.acquisition.exposure_us = self.exposure_spin.value()
        cfg.acquisition.gain_db = self.gain_spin.value()
        cfg.acquisition.binning = int(self.binning_combo.currentData() or 1)
        cfg.acquisition.roi.width = self.width_spin.value()
        cfg.acquisition.roi.height = self.height_spin.value()
        cfg.acquisition.pixel_format = self.pixfmt_combo.currentText().strip()
        cfg.acquisition.queue_size = self.queue_spin.value()

        cfg.session.output_dir = self.output_dir.text().strip() or "recordings"
        cfg.session.clip_name = self.clip_name.text().strip()
        cfg.session.filename_pattern = self.pattern_edit.text().strip() or \
            "{clip}_{serial}.mp4"
        cfg.session.duration_s = self.duration_spin.value()
        cfg.session.segment_minutes = self.segment_spin.value()
        cfg.encoder.ffmpeg = self.ffmpeg_path.text().strip() or "ffmpeg"
        cfg.encoder.codec = self.codec_combo.currentData()
        cfg.encoder.hardware = self.hw_combo.currentData()
        q = self.quality_spin.value()
        cfg.encoder.quality = None if q < 0 else q
        if cfg.encoder.codec == "custom":
            cfg.encoder.output_args = self.codec_args.text().split()

        for row in range(self.camera_table.rowCount()):
            check = self.camera_table.cellWidget(row, 0)
            serial = self.camera_table.item(row, 1).text()
            override = cfg.cameras.setdefault(serial, CameraOverride())
            override.enabled = check.isChecked()
        cfg.validate()
        return cfg

    def _sync_mode_changed(self) -> None:
        mode = self.sync_combo.currentData()
        hardware = mode == "hardware_trigger"
        for widget in (self.trigger_source, self.trigger_activation, self.master_serial):
            widget.setEnabled(hardware)

    @staticmethod
    def _set_combo(combo: QComboBox, data) -> None:
        i = combo.findData(data)
        if i >= 0:
            combo.setCurrentIndex(i)

    def _encoder_changed(self, *_args) -> None:
        """Show the ffmpeg arguments the settings produce; edit only for Custom."""
        codec = self.codec_combo.currentData()
        custom = codec == "custom"
        cpu_only = codec in ("mjpeg", "ffv1")
        self.hw_combo.setEnabled(not custom and not cpu_only)
        self.quality_spin.setEnabled(not custom and codec != "ffv1")
        self.test_encoder_btn.setEnabled(not custom)
        self.codec_args.setReadOnly(not custom)
        if custom:
            self.codec_args.setText(getattr(self, "_custom_args", ""))
            return
        hw = "cpu" if cpu_only else self.hw_combo.currentData()
        q = self.quality_spin.value()
        if hw == "auto":
            self.codec_args.setText("(chosen when recording starts: first working GPU, "
                                    "else CPU - press Test to see which)")
            return
        built = build_args(codec, hw, None if q < 0 else q,
                           container=container_of(self.pattern_edit.text() or ".mp4"))
        self.codec_args.setText(" ".join(built[0] + built[1]) if built else
                                f"{CODECS[codec]} is not available on {HARDWARE[hw]}")

    def _custom_args_edited(self, text: str) -> None:
        if self.codec_combo.currentData() == "custom":
            self._custom_args = text

    def _test_encoders(self) -> None:
        codec = self.codec_combo.currentData()
        if codec == "custom":
            return
        ffmpeg = self.ffmpeg_path.text().strip() or "ffmpeg"
        n = sum(1 for row in range(self.camera_table.rowCount())
                if self.camera_table.cellWidget(row, 0).isChecked()) or 1
        lines = []
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for hw in ["cpu"] + GPU_ORDER:
                if encoder_name(codec, hw) is None:
                    continue
                ok, msg = test_encoder(ffmpeg, codec, hw, 1 if hw == "cpu" else n)
                lines.append(f"{'OK' if ok else 'no'}  {HARDWARE[hw]}: {msg}")
                log.info("encoder test %s / %s: %s", codec, hw, msg)
        finally:
            QApplication.restoreOverrideCursor()
        QMessageBox.information(
            self, "Encoder test",
            f"{CODECS[codec]}, {n} parallel encode(s) per GPU:\n\n" + "\n".join(lines))

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Recording folder",
                                                self.output_dir.text())
        if path:
            self.output_dir.setText(path)

    # ---- recording name ----------------------------------------------------
    def _update_name_example(self, *_args) -> None:
        """Show where the next recording will be saved, or what is wrong with the name."""
        template = self.clip_name.text().strip() or DEFAULT_CLIP_NAME
        problem = check_clip_name(template)
        if problem:
            self.name_example.setText(f"⚠ {problem}")
            self.name_example.setStyleSheet("color: #f85149;")
            self.name_example.setToolTip("")
            return
        clip = expand_clip_name(template)
        folder = self.output_dir.text().strip() or "recordings"
        pattern = self.pattern_edit.text().strip() or "{clip}_{serial}.mp4"
        serial = self.cameras[0].info.serial if self.cameras else "12345678"
        try:
            filename = pattern.format(clip=clip, serial=serial, name=serial)
        except (KeyError, IndexError, ValueError):
            filename = pattern
        parts = [folder, clip]
        if self.segment_spin.value() > 0:
            parts.append("segment_0000")
        path = os.path.join(*parts, filename)
        # Zero-width spaces after the separators let the label wrap long paths.
        self.name_example.setText("Saved as: " + path.replace(os.sep, os.sep + "​")
                                  .replace("/", "/​"))
        self.name_example.setToolTip(path)
        self.name_example.setStyleSheet("color: #8b949e;")

    def _show_name_help(self) -> None:
        now = datetime.now()
        values = {k: now.strftime(f) for k, f in CLIP_VARIABLES.items()}
        examples = [DEFAULT_CLIP_NAME, "mouse3_{date}_{time}", "pilot_{date}",
                    "session2_trial1"]
        rows = "\n".join(f"    {e:<24} →  {e.format(**values)}" for e in examples)
        QMessageBox.information(
            self, "Naming a recording",
            "The recording name is used for the recording's folder and for its "
            "video files.\n\n"
            "You can type any name. These variables are replaced when you press "
            "Record:\n\n"
            f"    {{date}}    today's date, e.g. {values['date']} (year month day)\n"
            f"    {{time}}    the start time, e.g. {values['time']} "
            "(hours minutes seconds)\n\n"
            f"Examples:\n\n{rows}\n\n"
            "Not allowed in names:  \\ / : * ? \" < > |\n\n"
            "Nothing is ever overwritten: if a recording with the same name already "
            "exists, _2, _3, ... is added to the new one.\n\n"
            "The line under the name shows exactly where the files will be saved.")

    # ---- config files ------------------------------------------------------
    def load_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load config",
                                              os.path.dirname(self.default_path),
                                              "JSON files (*.json)")
        if not path:
            return
        try:
            self.cfg = RecorderConfig.load(path)
        except Exception as exc:
            QMessageBox.critical(self, "Could not load config", str(exc))
            return
        self._apply_config_to_widgets()
        log.info("loaded config from %s", path)
        self._schedule_preview_restart()

    def save_config(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save config",
                                              os.path.dirname(self.default_path),
                                              "JSON files (*.json)")
        if not path:
            return
        self._save_to(path)

    def save_default(self) -> None:
        answer = QMessageBox.question(
            self, "Save as default",
            f"Make the current settings the default?\n\nThey are written to\n"
            f"{self.default_path}\nand loaded every time the program starts.")
        if answer == QMessageBox.Yes:
            self._save_to(self.default_path)

    def _save_to(self, path: str) -> None:
        try:
            self._collect_config().save(path)
        except Exception as exc:
            QMessageBox.critical(self, "Could not save config", str(exc))
            return
        log.info("saved settings to %s", path)
        self.statusBar().showMessage(f"Settings saved to {path}", 5000)

    # ---- discovery ---------------------------------------------------------
    def discover_cameras(self) -> None:
        if self.session is not None or self._busy:
            return
        self._restart_timer.stop()
        try:
            cfg = copy.deepcopy(self._collect_config())
        except Exception as exc:
            QMessageBox.warning(self, "Invalid settings", str(exc))
            return
        live, old = self.live, self.cameras
        self.live, self.cameras = None, []
        self.preview.set_cameras([])
        self.camera_table.setRowCount(0)
        self.sync_label.setText("searching for cameras…")

        def job():
            if live is not None:
                live.stop()
            for cam in old:
                try:
                    cam.close()
                except Exception:
                    pass
            return self._discover_and_open(cfg)

        self._run_worker(job, self._on_discovered, self._on_discover_failed)

    @staticmethod
    def _discover_and_open(cfg):
        cams, backend = discover(cfg)
        for cam in cams:
            cam.open()
            cam.capabilities()          # cache while we are on the worker thread
        return cams, backend

    def _on_discovered(self, result) -> None:
        self.cameras, self._backend_name = result
        self.backend_label.setText(
            f"backend: {self._backend_name}   ({len(self.cameras)} cameras)")

        self.camera_table.setRowCount(len(self.cameras))
        self.master_serial.clear()
        self.master_serial.addItem("(external signal generator)")
        for row, cam in enumerate(self.cameras):
            check = QCheckBox()
            check.setChecked(self.cfg.override_for(cam.info.serial).enabled)
            check.toggled.connect(self._schedule_preview_restart)
            self.camera_table.setCellWidget(row, 0, check)
            self.camera_table.setItem(row, 1, QTableWidgetItem(cam.info.serial))
            self.camera_table.setItem(row, 2, QTableWidgetItem(cam.info.model))
            self.master_serial.addItem(cam.info.serial)
        if self.cfg.sync.hardware.master_serial:
            self.master_serial.setCurrentText(self.cfg.sync.hardware.master_serial)
        self._fit_camera_list()
        self._update_name_example()

        self._rebuild_stats_table([c.info.serial for c in self.cameras])
        if self.cameras:
            caps = set.intersection(*(c.capabilities() for c in self.cameras))
            self.sync_label.setText(
                f"{len(self.cameras)} cameras found · shared capabilities: "
                f"{', '.join(sorted(c.value for c in caps)) or 'none'}")
        else:
            self.sync_label.setText("no cameras found")
        log.info("discovered %d cameras via %s", len(self.cameras), self._backend_name)
        self._update_buttons()
        self._start_preview()

    def _on_discover_failed(self, message: str) -> None:
        self.sync_label.setText("camera search failed")
        self._update_buttons()
        log.error("camera search failed: %s", message)
        QMessageBox.critical(self, "Could not find cameras", message)

    def _fit_camera_list(self) -> None:
        """Make the camera list tall enough to show every camera, if there is room."""
        table = self.camera_table
        table.resizeRowsToContents()
        rows = sum(table.rowHeight(r) for r in range(table.rowCount()))
        needed_table = (table.horizontalHeader().height() + rows
                        + 2 * table.frameWidth() + 4)
        # Everything in the group except the table: title, margins, backend label.
        needed = (self.camera_box.sizeHint().height() - table.sizeHint().height()
                  + needed_table)
        total = sum(self.left_split.sizes())
        if total <= 0:
            return
        top = min(needed, int(total * 0.6))
        self.left_split.setSizes([top, total - top])

    def _rebuild_stats_table(self, serials: List[str]) -> None:
        self.stats_table.setRowCount(len(serials))
        for row, serial in enumerate(serials):
            self.stats_table.setItem(row, 0, QTableWidgetItem(serial))
            for col in range(1, 8):
                self.stats_table.setItem(row, col, QTableWidgetItem("-"))

    def _selected_cameras(self, cfg) -> List[CameraBackend]:
        return [c for c in self.cameras if cfg.override_for(c.info.serial).enabled]

    # ---- live preview ------------------------------------------------------
    def _schedule_preview_restart(self, *_args) -> None:
        if self._loading or not self.cameras or self.session is not None:
            return
        self._restart_timer.start()

    def _start_preview(self) -> None:
        """(Re)start the free-running preview with the current settings."""
        if self.session is not None or not self.cameras:
            return
        if self._busy:
            self._restart_timer.start()           # try again when the worker is done
            return
        try:
            cfg = copy.deepcopy(self._collect_config())
        except Exception as exc:
            log.warning("live preview not started: %s", exc)
            return
        selected = self._selected_cameras(cfg)
        old = self.live
        live = LivePreview(cfg, selected)
        self.live = live
        self.preview.set_cameras([(c.info.serial, c.info.serial) for c in selected])
        self.sync_label.setText("starting live preview…")

        def job():
            if old is not None:
                old.stop()
            live.start()

        self._run_worker(job, self._on_preview_started, self._on_preview_failed)

    def _on_preview_started(self, _result) -> None:
        self._update_buttons()
        if self.live is None:
            return
        # Settings changed while the cameras were starting.
        self._live_setting_changed()
        self.sync_label.setStyleSheet("")
        self.sync_label.setText(
            "Live preview (cameras not synchronised, nothing is saved) · "
            "press Record to start recording")

    def _on_preview_failed(self, message: str) -> None:
        self._update_buttons()
        log.error("live preview failed: %s", message)
        self.sync_label.setText("live preview failed - see the log")

    def _live_setting_changed(self, *_args) -> None:
        """Exposure, gain and frame rate go straight to the previewing cameras."""
        live = self.live
        if self._loading or live is None or not live.running:
            return
        try:
            cfg = copy.deepcopy(self._collect_config())
        except Exception:
            return
        live.apply_live(cfg)

    # ---- recording ---------------------------------------------------------
    def start_recording(self) -> None:
        if self.session is not None or self._busy:
            return
        try:
            cfg = copy.deepcopy(self._collect_config())
        except Exception as exc:
            QMessageBox.warning(self, "Invalid settings", str(exc))
            return
        problem = check_clip_name(cfg.session.clip_name or DEFAULT_CLIP_NAME)
        if problem:
            QMessageBox.warning(self, "Recording name",
                                f"Please change the recording name: {problem}.")
            return
        selected = self._selected_cameras(cfg)
        if not selected:
            QMessageBox.warning(self, "No cameras", "Select at least one camera.")
            return
        try:
            session = RecordingSession(cfg, selected)
        except Exception as exc:
            QMessageBox.warning(self, "Could not start recording", str(exc))
            return

        self._restart_timer.stop()
        live, self.live = self.live, None
        self.session = session
        self.preview.set_cameras([(c.info.serial, c.info.serial) for c in selected])
        self._rebuild_stats_table([c.info.serial for c in selected])
        log.info("arming %d cameras…", len(selected))
        self.sync_label.setText("preparing the recording…")

        def job():
            if live is not None:
                live.stop()
            session.prepare(progress=lambda m: log.info("%s", m))
            session.start()
            return session.out_dir

        self._run_worker(job, self._on_started, self._on_start_failed)

    def _on_started(self, out_dir) -> None:
        self._update_buttons()
        log.info("recording to %s", out_dir)

    def _on_start_failed(self, message: str) -> None:
        log.error("could not start recording: %s", message)
        session, self.session = self.session, None
        if session is not None:
            try:
                session.stop()
            except Exception:
                pass
        self._update_buttons()
        QMessageBox.critical(self, "Could not start recording", message)
        self._start_preview()

    def stop_recording(self) -> None:
        if self.session is None or self._busy:
            return
        log.info("stopping…")
        session = self.session
        self._run_worker(session.stop, self._on_stopped, self._on_stop_failed)

    def _on_stopped(self, status) -> None:
        self.session = None
        self._update_buttons()
        report = status.get("sync_report", {})
        written = sum(status.get("frames_written", {}).values())
        log.info("finished: %d frames written to %s", written, status.get("output_dir"))
        log.info("sync: mean %.3f ms, max %.3f ms over %d frame groups",
                 report.get("spread_ms_mean", 0), report.get("spread_ms_max", 0),
                 report.get("compared", 0))
        for serial, stats in status.get("cameras", {}).items():
            self.preview.update_stats(serial, stats, state="idle")
        self._update_name_example()          # {time} has moved on
        if status.get("errors"):
            QMessageBox.warning(self, "Recording finished with errors",
                                "\n".join(status["errors"][:10]))
        self._start_preview()

    def _on_stop_failed(self, message: str) -> None:
        self.session = None
        self._update_buttons()
        log.error("stopping the recording failed: %s", message)
        QMessageBox.critical(self, "Error while stopping", message)
        self._start_preview()

    # ---- polling -----------------------------------------------------------
    def _poll(self) -> None:
        if self.session is not None:
            # Not while it is being armed or stopped on the worker thread.
            if not self._busy and self.session.elapsed():
                self._poll_session()
            return
        live = self.live
        if live is None or not live.running:
            return
        for serial in list(self.preview.tiles):
            image = live.preview(serial)
            if image is not None:
                self.preview.update_image(serial, image)
            self.preview.update_stats(serial, live.stats(serial), state="live")

    def _poll_session(self) -> None:
        session = self.session
        status = session.status()
        elapsed = int(status["elapsed_s"])
        self.elapsed_label.setText(
            f"{elapsed // 3600:02d}:{elapsed % 3600 // 60:02d}:{elapsed % 60:02d}")

        for row in range(self.stats_table.rowCount()):
            serial = self.stats_table.item(row, 0).text()
            stats = status["cameras"].get(serial)
            if not stats:
                continue
            values = [f"{stats['fps']:.1f}", stats["grabbed"], stats["written"],
                      stats["dropped_queue"], stats["skipped_camera"],
                      stats.get("placeholders", 0), stats["queue_depth"]]
            for col, value in enumerate(values, start=1):
                item = self.stats_table.item(row, col)
                item.setText(str(value))
                if col in (4, 5, 6) and int(value) > 0:
                    item.setForeground(QColor("#f85149"))
            self.preview.update_stats(serial, stats)
            image = session.preview(serial)
            if image is not None:
                self.preview.update_image(serial, image)

        report = status["sync_report"]
        ok = report["spread_ms_max"] <= session.cfg.sync.max_spread_ms
        self.sync_label.setText(
            f"RECORDING · {status['sync_mode']} · {status['sync_detail']} · spread now "
            f"{report['spread_ms_last']:.3f} ms, max {report['spread_ms_max']:.3f} ms"
            f"{'' if ok else '  ⚠ OVER THRESHOLD'}")
        self.sync_label.setStyleSheet("" if ok else "color: #f85149; font-weight: bold;")
        self.rate_label.setText(f"{status['throughput_mb_s']:.1f} MB/s")
        try:
            free = shutil.disk_usage(status["output_dir"]).free
            self.disk_label.setText(f"{free / 1e9:.1f} GB free")
        except Exception:
            pass

        # The session stopped by itself (duration reached, disk full, ...).
        if not session.is_running() and not self._busy:
            self.stop_recording()

    # ---- plumbing ----------------------------------------------------------
    def _update_buttons(self) -> None:
        recording = self.session is not None
        idle = not recording and not self._busy
        self.btn_discover.setEnabled(idle)
        self.btn_record.setEnabled(idle and bool(self.cameras))
        self.btn_stop.setEnabled(recording and not self._busy
                                 and self.session.is_running())
        # Settings cannot change a running recording; make that obvious.
        self.settings_scroll.setEnabled(not recording)
        self.camera_table.setEnabled(not recording)
        self.config_buttons.setEnabled(not recording)

    def _run_worker(self, fn, on_ok, on_err) -> None:
        self._workers = [w for w in self._workers if not w.isFinished()]
        worker = Worker(fn, on_ok, on_err)
        worker.finished_ok.connect(self._worker_ok)
        worker.failed.connect(self._worker_failed)
        self._workers.append(worker)
        self._busy = True
        self._update_buttons()
        worker.start()

    def _worker_ok(self, worker: Worker, result) -> None:
        self._busy = False
        worker.on_ok(result)

    def _worker_failed(self, worker: Worker, message: str) -> None:
        self._busy = False
        worker.on_err(message)

    def closeEvent(self, event) -> None:
        if self.session is not None and self.session.is_running():
            answer = QMessageBox.question(
                self, "Recording in progress",
                "A recording is still running. Stop it and quit?")
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            try:
                self.session.stop()
            except Exception:
                log.exception("stop failed during shutdown")
        self.poll_timer.stop()
        self._restart_timer.stop()
        for worker in self._workers:
            worker.wait(30000)
        if self.live is not None:
            self.live.stop()
        for cam in self.cameras:
            try:
                cam.close()
            except Exception:
                pass
        self.cameras = []
        event.accept()


def apply_dark_palette(app: QApplication) -> None:
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#0d1117"))
    palette.setColor(QPalette.WindowText, QColor("#e6edf3"))
    palette.setColor(QPalette.Base, QColor("#161b22"))
    palette.setColor(QPalette.AlternateBase, QColor("#0d1117"))
    palette.setColor(QPalette.Text, QColor("#e6edf3"))
    palette.setColor(QPalette.Button, QColor("#21262d"))
    palette.setColor(QPalette.ButtonText, QColor("#e6edf3"))
    palette.setColor(QPalette.Highlight, QColor("#1f6feb"))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.ToolTipBase, QColor("#161b22"))
    palette.setColor(QPalette.ToolTipText, QColor("#e6edf3"))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor("#6e7681"))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#6e7681"))
    app.setPalette(palette)


def run_gui(config_path: Optional[str] = None) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    apply_dark_palette(app)
    window = MainWindow(config_path)
    window.show()
    return app.exec()
