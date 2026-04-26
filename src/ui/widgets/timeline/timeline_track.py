from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QColor, QPen, QPolygon, QMouseEvent, QKeyEvent
from PySide6.QtCore import Qt, Signal, QPoint
from typing import List, Set, Dict

class TimelineTrackWidget(QWidget):
    time_scrubbed = Signal(float)
    keyframe_selected = Signal(int)
    keyframes_mutated = Signal(dict)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(60)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.ClickFocus)
        
        self.duration_max: float = 10.0
        self.current_time: float = 0.0
        
        self.keyframes: List[float] = []
        self.selected_indices: Set[int] = set()
        
        self.is_scrubbing: bool = False
        self.is_box_selecting: bool = False
        self.box_start_x: float = 0.0
        self.box_end_x: float = 0.0
        
        self.drag_mode: str = "NONE" 
        self.drag_start_time: float = 0.0
        self.drag_initial_times: Dict[int, float] = {}
        self.ghost_keyframes: List[float] = []
        
        self.COLOR_BG = QColor(40, 40, 40)
        self.COLOR_GRID = QColor(80, 80, 80)
        self.COLOR_PLAYHEAD = QColor(66, 165, 245)
        self.COLOR_KF_NORMAL = QColor(180, 180, 180)
        self.COLOR_KF_SELECTED = QColor(255, 165, 0)
        self.COLOR_KF_BASE = QColor(231, 76, 60)
        self.COLOR_BOX_FILL = QColor(66, 165, 245, 50)
        self.COLOR_BOX_BORDER = QColor(66, 165, 245, 150)
        
    def set_max_time(self, t: float) -> None:
        self.duration_max = max(0.1, t)
        self.update()
        
    def set_time(self, t: float) -> None:
        self.current_time = max(0.0, min(self.duration_max, t))
        self.update()
        
    def set_keyframes(self, kf_times: List[float]) -> None:
        self.keyframes = kf_times
        self.selected_indices = {i for i in self.selected_indices if i < len(self.keyframes)}
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
        
        step_sec = 1.0
        if self.duration_max > 20.0: step_sec = 5.0
        if self.duration_max < 2.0: step_sec = 0.1
        
        num_ticks = int(self.duration_max / step_sec) + 1
        for i in range(num_ticks):
            t = i * step_sec
            x = self._time_to_x(t)
            painter.drawLine(int(x), 0, int(x), h)
            
        sub_ticks = int(self.duration_max / (step_sec / 5)) + 1
        for i in range(sub_ticks):
            t = i * (step_sec / 5)
            x = self._time_to_x(t)
            painter.drawLine(int(x), 0, int(x), 5)
            
        y_center = h // 2
        
        for t in self.ghost_keyframes:
            x = int(self._time_to_x(t))
            painter.setBrush(QColor(255, 165, 0, 100))
            painter.setPen(Qt.NoPen)
            size = 12
            poly = QPolygon([
                QPoint(x, y_center - size // 2), QPoint(x + size // 2, y_center),
                QPoint(x, y_center + size // 2), QPoint(x - size // 2, y_center)
            ])
            painter.drawPolygon(poly)

        for i, kf_time in enumerate(self.keyframes):
            x = int(self._time_to_x(kf_time))
            is_selected = (i in self.selected_indices)
            
            if is_selected: color = self.COLOR_KF_SELECTED
            elif i == 0: color = self.COLOR_KF_BASE
            else: color = self.COLOR_KF_NORMAL
            
            painter.setBrush(color)
            painter.setPen(Qt.NoPen)
            
            size = 12
            poly = QPolygon([
                QPoint(x, y_center - size // 2), QPoint(x + size // 2, y_center),
                QPoint(x, y_center + size // 2), QPoint(x - size // 2, y_center)
            ])
            painter.drawPolygon(poly)
            
        if self.is_box_selecting:
            rx = min(self.box_start_x, self.box_end_x)
            rw = abs(self.box_start_x - self.box_end_x)
            painter.fillRect(int(rx), 0, int(rw), h, self.COLOR_BOX_FILL)
            painter.setPen(QPen(self.COLOR_BOX_BORDER, 1))
            painter.drawRect(int(rx), 0, int(rw), h - 1)
            
        x_playhead = int(self._time_to_x(self.current_time))
        painter.setPen(QPen(self.COLOR_PLAYHEAD, 2))
        painter.drawLine(x_playhead, 0, x_playhead, h)
        
        painter.setBrush(self.COLOR_PLAYHEAD)
        painter.setPen(Qt.NoPen)
        cap = QPolygon([
            QPoint(x_playhead - 6, 0), QPoint(x_playhead + 6, 0), QPoint(x_playhead, 8)
        ])
        painter.drawPolygon(cap)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            x = event.position().x()
            clicked_kf = self._get_kf_at_pos(x)
            
            if clicked_kf != -1:
                if event.modifiers() & Qt.ShiftModifier:
                    if clicked_kf in self.selected_indices:
                        self.selected_indices.remove(clicked_kf)
                    else:
                        self.selected_indices.add(clicked_kf)
                else:
                    if clicked_kf not in self.selected_indices:
                        self.selected_indices = {clicked_kf}
                
                rep_idx = list(self.selected_indices)[0] if self.selected_indices else -1
                self.keyframe_selected.emit(rep_idx)
                
                self.drag_initial_times = {i: self.keyframes[i] for i in self.selected_indices if i > 0}
                if self.drag_initial_times:
                    self.drag_start_time = self._x_to_time(x)
                    if event.modifiers() & Qt.AltModifier:
                        self.drag_mode = "COPY"
                    elif event.modifiers() & Qt.ControlModifier:
                        self.drag_mode = "SCALE"
                    else:
                        self.drag_mode = "MOVE"
            else:
                if not (event.modifiers() & Qt.ShiftModifier):
                    self.selected_indices.clear()
                    self.keyframe_selected.emit(-1)
                
                self.is_box_selecting = True
                self.box_start_x = x
                self.box_end_x = x
                
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        x = event.position().x()
        
        if self.is_box_selecting:
            self.box_end_x = x
            self.update()
        elif self.drag_mode != "NONE":
            t = self._x_to_time(x)
            if self.drag_mode == "MOVE":
                delta = t - self.drag_start_time
                for i, init_t in self.drag_initial_times.items():
                    self.keyframes[i] = max(0.01, init_t + delta)
            elif self.drag_mode == "SCALE":
                origin = min(self.drag_initial_times.values())
                dist = self.drag_start_time - origin
                factor = (t - origin) / dist if dist > 0.01 else 1.0
                for i, init_t in self.drag_initial_times.items():
                    self.keyframes[i] = max(0.01, origin + (init_t - origin) * factor)
            elif self.drag_mode == "COPY":
                delta = t - self.drag_start_time
                self.ghost_keyframes = [max(0.01, init_t + delta) for init_t in self.drag_initial_times.values()]
            self.update()
        else:
            self.is_scrubbing = True
            new_time = self._x_to_time(x)
            self.time_scrubbed.emit(new_time)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            if self.is_box_selecting:
                min_x = min(self.box_start_x, self.box_end_x)
                max_x = max(self.box_start_x, self.box_end_x)
                for i, kf_time in enumerate(self.keyframes):
                    if min_x <= self._time_to_x(kf_time) <= max_x:
                        self.selected_indices.add(i)
                self.is_box_selecting = False
                
            elif self.drag_mode != "NONE":
                payload = {}
                if self.drag_mode in ["MOVE", "SCALE"]:
                    payload = {"mode": "UPDATE", "data": {i: self.keyframes[i] for i in self.drag_initial_times}}
                elif self.drag_mode == "COPY":
                    delta = self._x_to_time(event.position().x()) - self.drag_start_time
                    payload = {"mode": "COPY", "indices": list(self.drag_initial_times.keys()), "offset": delta}
                    self.selected_indices.clear() 
                    
                if payload:
                    self.keyframes_mutated.emit(payload)
                    
                self.drag_mode = "NONE"
                self.drag_initial_times.clear()
                self.ghost_keyframes.clear()
                
            self.is_scrubbing = False
            self.update()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            valid_indices = [i for i in self.selected_indices if i > 0]
            if valid_indices:
                self.keyframes_mutated.emit({"mode": "DELETE_BULK", "indices": valid_indices})
                self.selected_indices.clear()
                self.update()
        else:
            super().keyPressEvent(event)