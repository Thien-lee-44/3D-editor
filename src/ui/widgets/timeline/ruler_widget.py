"""
Timeline Ruler Widget.

Provides a horizontal ruler for displaying time increments, subdivisions, 
and the current playback head (playhead) position.
"""

from typing import Any
from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QColor, QPen, QPolygon, QMouseEvent, QFont
from PySide6.QtCore import Qt, Signal, QPoint


class RulerWidget(QWidget):
    """
    Dedicated horizontal ruler for displaying time increments and the playhead.
    Separating this from the track allows for independent vertical scrolling of tracks later.
    """
    time_scrubbed = Signal(float)

    def __init__(self) -> None:
        super().__init__()
        self.setFixedHeight(28)
        self.setMouseTracking(True)
        
        self.duration_max: float = 10.0
        self.current_time: float = 0.0
        self.is_scrubbing: bool = False
        
        self.COLOR_BG = QColor(45, 45, 45)
        self.COLOR_GRID = QColor(100, 100, 100)
        self.COLOR_TEXT = QColor(200, 200, 200)
        self.COLOR_PLAYHEAD = QColor(66, 165, 245)

    def set_max_time(self, t: float) -> None:
        """Updates the maximum visible timeline duration."""
        self.duration_max = max(0.1, t)
        self.update()
        
    def set_time(self, t: float) -> None:
        """Sets the current playhead time and forces a repaint."""
        self.current_time = max(0.0, min(self.duration_max, t))
        self.update()

    def _time_to_x(self, t: float) -> float:
        """Translates a logical time value into a physical pixel X-coordinate."""
        return (t / self.duration_max) * self.width()

    def _x_to_time(self, x: float) -> float:
        """Translates a physical pixel X-coordinate back into a logical time value."""
        t = (x / self.width()) * self.duration_max
        return max(0.0, min(self.duration_max, t))

    def paintEvent(self, event: Any) -> None:
        """Renders the ruler background, tick marks, text labels, and the playhead."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        painter.fillRect(0, 0, w, h, self.COLOR_BG)
        painter.setPen(QPen(self.COLOR_GRID, 1))
        painter.setFont(QFont("Arial", 8))
        
        # Determine logical step size based on zoom/duration level
        step_sec = 1.0
        if self.duration_max > 20.0: 
            step_sec = 5.0
        if self.duration_max < 2.0: 
            step_sec = 0.1
        
        # Draw primary ticks and text labels
        num_ticks = int(self.duration_max / step_sec) + 1
        for i in range(num_ticks):
            t = i * step_sec
            x = self._time_to_x(t)
            painter.drawLine(int(x), h - 10, int(x), h)
            painter.setPen(QPen(self.COLOR_TEXT))
            painter.drawText(int(x) + 4, h - 10, f"{t:.1f}s")
            painter.setPen(QPen(self.COLOR_GRID, 1))
            
        # Draw subdivision ticks
        sub_ticks = int(self.duration_max / (step_sec / 5)) + 1
        for i in range(sub_ticks):
            t = i * (step_sec / 5)
            x = self._time_to_x(t)
            painter.drawLine(int(x), h - 4, int(x), h)
            
        # Draw Playhead cap
        x_playhead = int(self._time_to_x(self.current_time))
        painter.setBrush(self.COLOR_PLAYHEAD)
        painter.setPen(Qt.NoPen)
        cap = QPolygon([
            QPoint(x_playhead - 6, 0),
            QPoint(x_playhead + 6, 0),
            QPoint(x_playhead + 6, h - 6),
            QPoint(x_playhead, h),
            QPoint(x_playhead - 6, h - 6)
        ])
        painter.drawPolygon(cap)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Initiates time scrubbing when the user clicks the ruler."""
        if event.button() == Qt.LeftButton:
            self.is_scrubbing = True
            self.time_scrubbed.emit(self._x_to_time(event.position().x()))
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Emits scrub signals as the user drags along the ruler."""
        if self.is_scrubbing:
            self.time_scrubbed.emit(self._x_to_time(event.position().x()))

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Terminates time scrubbing when the mouse is released."""
        if event.button() == Qt.LeftButton:
            self.is_scrubbing = False