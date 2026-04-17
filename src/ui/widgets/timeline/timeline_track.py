from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QColor, QPen, QPolygon, QMouseEvent, QFont
from PySide6.QtCore import Qt, Signal, QPoint
from typing import List

class TimelineTrackWidget(QWidget):
    time_scrubbed = Signal(float)
    keyframe_selected = Signal(int)
    keyframe_moved = Signal(int, float)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(60)
        self.setMouseTracking(True)
        
        self.duration_max: float = 10.0
        self.current_time: float = 0.0
        
        self.keyframes: List[float] = []
        self.selected_kf_index: int = -1
        
        self.is_scrubbing: bool = False
        self.is_dragging_kf: bool = False
        self.drag_kf_index: int = -1
        
        self.COLOR_BG = QColor(40, 40, 40)
        self.COLOR_GRID = QColor(80, 80, 80)
        self.COLOR_TEXT = QColor(150, 150, 150)
        self.COLOR_PLAYHEAD = QColor(66, 165, 245)
        self.COLOR_KF_NORMAL = QColor(180, 180, 180)
        self.COLOR_KF_SELECTED = QColor(255, 165, 0)
        self.COLOR_KF_BASE = QColor(231, 76, 60) 
        
    def set_max_time(self, t: float) -> None:
        self.duration_max = max(0.1, t)
        self.update()
        
    def set_time(self, t: float) -> None:
        self.current_time = max(0.0, min(self.duration_max, t))
        self.update()
        
    def set_keyframes(self, kf_times: List[float]) -> None:
        self.keyframes = kf_times
        self.update()

    def _time_to_x(self, t: float) -> float:
        return (t / self.duration_max) * self.width()

    def _x_to_time(self, x: float) -> float:
        t = (x / self.width()) * self.duration_max
        return max(0.0, min(self.duration_max, t))

    def _get_kf_at_pos(self, x: float) -> int:
        threshold = 8 
        for i, kf_time in enumerate(self.keyframes):
            kf_x = self._time_to_x(kf_time)
            if abs(x - kf_x) <= threshold:
                return i
        return -1

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        painter.fillRect(0, 0, w, h, self.COLOR_BG)
        painter.setPen(QPen(self.COLOR_GRID, 1))
        painter.setFont(QFont("Arial", 8))
        
        step_sec = 1.0
        if self.duration_max > 20.0: step_sec = 5.0
        if self.duration_max < 2.0: step_sec = 0.1
        
        num_ticks = int(self.duration_max / step_sec) + 1
        for i in range(num_ticks):
            t = i * step_sec
            x = self._time_to_x(t)
            painter.drawLine(int(x), 0, int(x), h)
            
            painter.setPen(QPen(self.COLOR_TEXT))
            painter.drawText(int(x) + 4, 12, f"{t:.1f}s")
            painter.setPen(QPen(self.COLOR_GRID, 1))
            
        sub_ticks = int(self.duration_max / (step_sec / 5)) + 1
        for i in range(sub_ticks):
            t = i * (step_sec / 5)
            x = self._time_to_x(t)
            painter.drawLine(int(x), 0, int(x), 5)
            
        y_center = h // 2
        for i, kf_time in enumerate(self.keyframes):
            x = int(self._time_to_x(kf_time))
            is_selected = (i == self.selected_kf_index)
            
            if is_selected:
                color = self.COLOR_KF_SELECTED
            elif i == 0:
                color = self.COLOR_KF_BASE
            else:
                color = self.COLOR_KF_NORMAL
            
            painter.setBrush(color)
            painter.setPen(Qt.NoPen)
            
            size = 12
            poly = QPolygon([
                QPoint(x, y_center - size // 2),
                QPoint(x + size // 2, y_center),
                QPoint(x, y_center + size // 2),
                QPoint(x - size // 2, y_center)
            ])
            painter.drawPolygon(poly)
            
        x_playhead = int(self._time_to_x(self.current_time))
        painter.setPen(QPen(self.COLOR_PLAYHEAD, 2))
        painter.drawLine(x_playhead, 0, x_playhead, h)
        
        painter.setBrush(self.COLOR_PLAYHEAD)
        painter.setPen(Qt.NoPen)
        cap = QPolygon([
            QPoint(x_playhead - 6, 0),
            QPoint(x_playhead + 6, 0),
            QPoint(x_playhead, 8)
        ])
        painter.drawPolygon(cap)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            x = event.position().x()
            clicked_kf = self._get_kf_at_pos(x)
            
            if clicked_kf != -1:
                self.selected_kf_index = clicked_kf
                self.keyframe_selected.emit(clicked_kf)
                
                self.is_dragging_kf = True
                self.drag_kf_index = clicked_kf
            else:
                self.selected_kf_index = -1
                self.keyframe_selected.emit(-1) 
                
                self.is_scrubbing = True
                new_time = self._x_to_time(x)
                self.time_scrubbed.emit(new_time)
                
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        x = event.position().x()
        
        if self.is_scrubbing:
            new_time = self._x_to_time(x)
            self.time_scrubbed.emit(new_time)
            
        # [DRAG LOCK]: Ngăn chặn việc vô tình kéo (drag) Base State đi khỏi mốc 0.0s
        elif self.is_dragging_kf and self.drag_kf_index > 0:
            new_time = self._x_to_time(x)
            if new_time > 0.01: 
                self.keyframes[self.drag_kf_index] = new_time
                self.keyframe_moved.emit(self.drag_kf_index, new_time)
                self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            # [DRAG LOCK]: Ngăn chặn việc vô tình kéo (drag) Base State đi khỏi mốc 0.0s
            if self.is_dragging_kf and self.drag_kf_index > 0:
                final_time = self._x_to_time(event.position().x())
                if final_time > 0.01:
                    self.keyframe_moved.emit(self.drag_kf_index, final_time)
                
            self.is_scrubbing = False
            self.is_dragging_kf = False
            self.drag_kf_index = -1