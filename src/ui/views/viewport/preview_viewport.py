from typing import Any, Dict, Tuple

from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import (QImage, QPixmap, QPainter, QPen, QColor, 
                           QFont, QResizeEvent)

from src.app import ctx

class AspectRatioContainer(QWidget):
    def __init__(self, child_widget: QWidget) -> None:
        super().__init__()
        self.setStyleSheet("background-color: #000;")
        self.child_widget = child_widget
        self.child_widget.setParent(self)
        self.target_w = 640
        self.target_h = 640
        
        self.setMinimumSize(1, 1)
        self.child_widget.setMinimumSize(1, 1)

    def set_target_resolution(self, w: int, h: int) -> None:
        self.target_w = max(1, w)
        self.target_h = max(1, h)
        self._update_layout()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_layout()

    def _update_layout(self) -> None:
        cw = self.width()
        ch = self.height()
        if cw <= 0 or ch <= 0: return
        
        aspect = self.target_w / float(self.target_h)
        container_aspect = cw / float(ch)
        
        if container_aspect > aspect:
            th = ch
            tw = int(th * aspect)
        else:
            tw = cw
            th = int(tw / aspect)
            
        ox = (cw - tw) // 2
        oy = (ch - th) // 2
        
        self.child_widget.setGeometry(ox, oy, tw, th)

class PreviewGLWidget(QWidget):
    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.payload: Dict[str, Any] = {}
        self.cpu_pixmap: QPixmap = None
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("background-color: #0d0d0d;")

    def update_frame(self, payload: Dict[str, Any], cpu_pixmap: QPixmap = None) -> None:
        self.payload = payload
        self.cpu_pixmap = cpu_pixmap
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        if self.cpu_pixmap:
            draw_w, draw_h = self.width(), self.height()
            scale_mode = Qt.FastTransformation if self.payload.get("is_live", False) else Qt.SmoothTransformation
            scaled = self.cpu_pixmap.scaled(draw_w, draw_h, Qt.IgnoreAspectRatio, scale_mode)
            painter.drawPixmap(0, 0, scaled)
        else:
            painter.setPen(QColor(100, 100, 100))
            painter.setFont(QFont("Arial", 10))
            painter.drawText(self.rect(), Qt.AlignCenter, "No preview data available.\nPress 'Live Preview' to start.")

        if self.payload:
            src_w = self.payload.get("width", self.width())
            src_h = self.payload.get("height", self.height())
            self._draw_overlays(painter, self.width(), self.height(), src_w, src_h)

        painter.end()

    def _draw_overlays(self, painter: QPainter, draw_w: int, draw_h: int, res_w: int, res_h: int) -> None:
        if not self.payload.get("show_bbox", True):
            return

        gt_objects = self.payload.get("objects", [])
        scale_x = draw_w / float(max(res_w, 1))
        scale_y = draw_h / float(max(res_h, 1))

        for obj in gt_objects:
            raw_name = obj.get("class_name", obj.get("entity_name", "Unknown"))
            clean_group_name = str(raw_name).split('/')[0].split('\\')[0]
            track_id = obj.get("track_id", -1)

            bbox = obj.get("bbox", obj.get("bbox_xyxy", [0, 0, 0, 0]))
            if not bbox or len(bbox) < 4: continue
            
            x1, y1, x2, y2 = bbox
            x1, x2 = x1 * scale_x, x2 * scale_x
            y1, y2 = y1 * scale_y, y2 * scale_y

            painter.setBrush(Qt.NoBrush) 
            painter.setPen(QPen(QColor(0, 255, 0), 2))
            painter.drawRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            
            label_text = f"{clean_group_name}#{track_id}"
            painter.setFont(QFont("Arial", 9, QFont.Bold))
            fm = painter.fontMetrics()
            label_w = max(90, fm.horizontalAdvance(label_text) + 10)

            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(0, 255, 0, 180))
            painter.drawRect(int(x1), int(y1) - 18, label_w, 18)
            
            painter.setPen(QColor(0, 0, 0))
            painter.drawText(int(x1) + 4, int(y1) - 4, label_text)

class PreviewViewportWindow(QWidget):
    def __init__(self, controller: Any) -> None:
        super().__init__()
        self._controller = controller
        self.setMinimumSize(1, 1)
        self.setup_ui()

    def setup_ui(self) -> None:
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        self.display_frame = PreviewGLWidget()
        self.display_frame.setMinimumSize(1, 1)
        
        self.canvas_container = AspectRatioContainer(self.display_frame)
        self.main_layout.addWidget(self.canvas_container, stretch=1)

    def get_resolution(self) -> Tuple[int, int]:
        gen_ctrl = getattr(self._controller, 'generator_ctrl', getattr(self, '_controller', None))
        if gen_ctrl and hasattr(gen_ctrl, 'view'):
            settings = gen_ctrl.view.get_settings()
            return settings.get("res_w", 640), settings.get("res_h", 640)
        return 640, 640
        
    def get_preview_mode(self) -> str:
        if hasattr(ctx, 'main_window') and hasattr(ctx.main_window, 'cmb_preview_mode'):
            return ctx.main_window.cmb_preview_mode.currentText()
        return "RGB"

    def is_bbox_enabled(self) -> bool:
        if hasattr(ctx, 'main_window') and hasattr(ctx.main_window, 'chk_preview_bbox'):
            return ctx.main_window.chk_preview_bbox.isChecked()
        return True

    def update_frame(self, payload: Dict[str, Any]) -> None:
        if not payload: return
        
        payload["show_bbox"] = self.is_bbox_enabled()
        mode = self.get_preview_mode()
        payload["mode"] = mode
        w, h = payload["width"], payload["height"]
        
        native_pixmap = None
        pixel_bytes = payload.get("modes", {}).get(mode)
        
        if pixel_bytes:
            native_pixmap = self._build_pixmap_from_bytes(pixel_bytes, w, h, mode)
            
        if self.canvas_container.target_w != w or self.canvas_container.target_h != h:
            self.canvas_container.set_target_resolution(w, h)
            
        self.display_frame.update_frame(payload, native_pixmap)

    def _build_pixmap_from_bytes(self, pixel_bytes: bytes, w: int, h: int, mode: str) -> QPixmap:
        if w <= 0 or h <= 0:
            return None

        if mode == "DEPTH":
            expected_size = w * h
            if len(pixel_bytes) != expected_size:
                return None
            qimg = QImage(pixel_bytes, w, h, w, QImage.Format_Grayscale8)
        else:
            expected_size = w * h * 3
            if len(pixel_bytes) != expected_size:
                return None
            qimg = QImage(pixel_bytes, w, h, w * 3, QImage.Format_RGB888)

        if qimg.isNull():
            return None

        safe_img = qimg.copy().mirrored(False, True)
        return QPixmap.fromImage(safe_img)