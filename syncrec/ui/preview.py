"""Live preview widgets.

The grid lays out however many cameras there are, so 2 or 16 look equally sensible.
Each tile draws the latest thumbnail the session produced plus the numbers you need
while a recording is running: frame rate, dropped frames, and whether that camera is
still delivering.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QGridLayout, QSizePolicy, QWidget

OK_COLOUR = QColor("#3fb950")
WARN_COLOUR = QColor("#d29922")
BAD_COLOUR = QColor("#f85149")
IDLE_COLOUR = QColor("#6e7681")
LIVE_COLOUR = QColor("#388bfd")


def numpy_to_qimage(array: np.ndarray) -> Optional[QImage]:
    """Wrap a contiguous uint8 array as a QImage (copied, so the source may change)."""
    if array is None or array.size == 0:
        return None
    if array.dtype != np.uint8:
        array = np.clip(array >> 8 if array.dtype == np.uint16 else array,
                        0, 255).astype(np.uint8)
    array = np.ascontiguousarray(array)
    if array.ndim == 2:
        h, w = array.shape
        fmt, stride = QImage.Format_Grayscale8, w
    elif array.ndim == 3 and array.shape[2] == 3:
        h, w, _ = array.shape
        fmt, stride = QImage.Format_BGR888, 3 * w
    elif array.ndim == 3 and array.shape[2] == 4:
        h, w, _ = array.shape
        fmt, stride = QImage.Format_RGBA8888, 4 * w
    else:
        return None
    return QImage(array.data, w, h, stride, fmt).copy()


class CameraTile(QWidget):
    """One camera: image, label, and a status strip."""

    def __init__(self, serial: str, title: str = "", parent=None):
        super().__init__(parent)
        self.serial = serial
        self.title = title or serial
        self._pixmap: Optional[QPixmap] = None
        self._stats: Dict[str, object] = {}
        self._state = "idle"
        self.setMinimumSize(220, 180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_image(self, array: np.ndarray) -> None:
        image = numpy_to_qimage(array)
        if image is not None:
            self._pixmap = QPixmap.fromImage(image)
            self.update()

    def set_stats(self, stats: Dict[str, object], state: str = "recording") -> None:
        self._stats = stats or {}
        self._state = state
        self.update()

    def clear(self) -> None:
        self._pixmap = None
        self._stats = {}
        self._state = "idle"
        self.update()

    def _status_colour(self) -> QColor:
        if self._state == "idle":
            return IDLE_COLOUR
        if self._stats.get("error"):
            return BAD_COLOUR
        if self._state == "live":
            return LIVE_COLOUR
        if int(self._stats.get("dropped_queue", 0) or 0) > 0:
            return WARN_COLOUR
        return OK_COLOUR

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        rect = self.rect()
        painter.fillRect(rect, QColor("#0d1117"))

        strip_h = 34
        image_rect = QRect(rect.x(), rect.y(), rect.width(), rect.height() - strip_h)
        if self._pixmap is not None:
            scaled = self._pixmap.scaled(image_rect.size(), Qt.KeepAspectRatio,
                                         Qt.SmoothTransformation)
            x = image_rect.x() + (image_rect.width() - scaled.width()) // 2
            y = image_rect.y() + (image_rect.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        else:
            painter.setPen(QPen(IDLE_COLOUR))
            painter.drawText(image_rect, Qt.AlignCenter, "no image")

        strip = QRect(rect.x(), rect.bottom() - strip_h, rect.width(), strip_h)
        painter.fillRect(strip, QColor("#161b22"))
        painter.fillRect(QRect(strip.x(), strip.y(), 4, strip.height()),
                         self._status_colour())

        font = QFont(painter.font())
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(QColor("#e6edf3")))
        painter.drawText(strip.adjusted(12, 0, -8, -strip_h // 2), Qt.AlignVCenter,
                         self.title)

        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QPen(QColor("#8b949e")))
        if self._state == "live" and self._stats:
            detail = (str(self._stats["error"])[:60] if self._stats.get("error") else
                      f"live preview   {float(self._stats.get('fps', 0)):.1f} fps")
        elif self._stats:
            detail = (f"{float(self._stats.get('fps', 0)):.1f} fps   "
                      f"{int(self._stats.get('written', 0))} written")
            dropped = int(self._stats.get("dropped_queue", 0) or 0)
            skipped = int(self._stats.get("skipped_camera", 0) or 0)
            if dropped:
                detail += f"   {dropped} dropped"
            if skipped:
                detail += f"   {skipped} missed"
            if self._stats.get("error"):
                detail = str(self._stats["error"])[:60]
        else:
            detail = "idle"
        painter.drawText(strip.adjusted(12, strip_h // 2, -8, 0), Qt.AlignVCenter, detail)
        painter.end()


class PreviewGrid(QWidget):
    """Arranges camera tiles in as square a grid as the count allows."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QGridLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(6)
        self.tiles: Dict[str, CameraTile] = {}

    def set_cameras(self, entries: List[tuple]) -> None:
        """entries: [(serial, title), ...] -- rebuilds the grid."""
        for tile in self.tiles.values():
            self._layout.removeWidget(tile)
            tile.deleteLater()
        self.tiles.clear()
        if not entries:
            return
        cols = max(1, math.ceil(math.sqrt(len(entries))))
        for i, (serial, title) in enumerate(entries):
            tile = CameraTile(serial, title, self)
            self.tiles[serial] = tile
            self._layout.addWidget(tile, i // cols, i % cols)

    def update_image(self, serial: str, array: np.ndarray) -> None:
        tile = self.tiles.get(serial)
        if tile is not None:
            tile.set_image(array)

    def update_stats(self, serial: str, stats: Dict[str, object],
                     state: str = "recording") -> None:
        tile = self.tiles.get(serial)
        if tile is not None:
            tile.set_stats(stats, state)

    def clear_all(self) -> None:
        for tile in self.tiles.values():
            tile.clear()
