from typing import Any, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import (QCursor, QDragEnterEvent, QDragMoveEvent,
                           QDropEvent, QFocusEvent, QKeyEvent, QMouseEvent,
                           QWheelEvent, QPainter, QPen, QColor, QFont)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QLabel, QMenu, QMessageBox
from OpenGL.GL import (glBindVertexArray, glUseProgram, glDisable,
                       GL_DEPTH_TEST, GL_CULL_FACE)

from src.app import AppEvent, ctx
from src.app.config import (CONTEXT_MENU_STYLE, DEFAULT_BG_COLOR,
                            TEXTURE_CHANNELS, VIEWPORT_HUD_STYLE)

class MainViewportView(QOpenGLWidget):
    """
    Main 3D viewport (Dumb View).
    Delegates rendering to the Engine and handles peripheral events.
    Includes QPainter overlays for Mathematical Ground Truth and Real-time AI Inference.
    """
    def __init__(self, controller: Any, parent: Optional[Any] = None) -> None:
        super().__init__(parent)
        self._controller = controller

        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.bg_color = DEFAULT_BG_COLOR

        # Overlay Flags
        self.show_debug_bboxes = False
        
        # Live AI Integration
        self.live_ai_engine = None
        self.show_ai_preview = False

        # HUD Orientation Labels
        self.lbl_x = QLabel("X", self)
        self.lbl_y = QLabel("Y", self)
        self.lbl_z = QLabel("Z", self)
        self.lbl_nx = QLabel("-X", self)
        self.lbl_ny = QLabel("-Y", self)
        self.lbl_nz = QLabel("-Z", self)
        self.labels_dict = {
            'X': self.lbl_x, 'Y': self.lbl_y, 'Z': self.lbl_z,
            '-X': self.lbl_nx, '-Y': self.lbl_ny, '-Z': self.lbl_nz
        }

    # =========================================================================
    # OPENGL LIFECYCLE
    # =========================================================================

    def initializeGL(self) -> None:
        ctx.engine.init_viewport_gl()

    def resizeGL(self, w: int, h: int) -> None:
        ctx.engine.resize_gl(w, h)

    def paintGL(self) -> None:
        active_axis = self._controller.active_axis
        hovered_axis = self._controller.hovered_axis
        hovered_screen_axis = self._controller.hovered_screen_axis

        # 1. Execute 3D Scene Rendering
        ctx.engine.render_viewport(
            self.width(), self.height(), self.bg_color,
            active_axis, hovered_axis, hovered_screen_axis
        )

        # 2. Update 3D Axis HUD
        self._update_hud_labels(hovered_screen_axis)

        # 3. Execute 2D Overlays
        if self.show_debug_bboxes or self.show_ai_preview:
            self._render_2d_overlays()

    def _update_hud_labels(self, hovered_screen_axis: str) -> None:
        for lbl in self.labels_dict.values():
            lbl.hide()

        for data in ctx.engine.get_screen_axis_labels_data(self.width(), self.height()):
            lbl = self.labels_dict[data['name']]
            if hovered_screen_axis and lbl.text() == hovered_screen_axis:
                lbl.setStyleSheet("color: #ffff00; font-weight: bold; background: transparent;")
            else:
                if lbl == self.lbl_x:
                    lbl.setStyleSheet("color: #ff3333; font-weight: bold; background: transparent;")
                elif lbl == self.lbl_y:
                    lbl.setStyleSheet("color: #33ff33; font-weight: bold; background: transparent;")
                elif lbl == self.lbl_z:
                    lbl.setStyleSheet("color: #3388ff; font-weight: bold; background: transparent;")
                else:
                    lbl.setStyleSheet(VIEWPORT_HUD_STYLE)

            lbl.move(data['x'], data['y'])
            lbl.show()
            lbl.raise_()

    def _render_2d_overlays(self) -> None:
        """
        Safely unbinds OpenGL states and renders Math-based Bounding Boxes (Green) 
        and AI-predicted Bounding Boxes (Blue) concurrently.
        """
        # Fetch raw RGB bytes BEFORE unbinding OpenGL states if AI is active
        ai_predictions = []
        if self.show_ai_preview and self.live_ai_engine:
            rgb_pixels = ctx.engine.renderer.capture_fbo_frame(ctx.engine.scene, self.width(), self.height(), mode="RGB")
            ai_predictions = self.live_ai_engine.predict_from_fbo(rgb_pixels, self.width(), self.height())

        glBindVertexArray(0)
        glUseProgram(0)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)

        painter = QPainter(self)
        
        # 1. Draw Mathematical Ground Truth (Green)
        if self.show_debug_bboxes:
            painter.setRenderHint(QPainter.Antialiasing, False)
            pen = QPen(QColor(0, 255, 0, 255))
            pen.setWidth(2)
            pen.setStyle(Qt.SolidLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)

            bboxes = ctx.engine.get_debug_bboxes(self.width(), self.height())
            for xmin, ymin, xmax, ymax in bboxes:
                x, y = int(xmin), int(ymin)
                w, h = int(xmax - xmin), int(ymax - ymin)
                painter.drawLine(x, y, x + w, y)         
                painter.drawLine(x + w, y, x + w, y + h) 
                painter.drawLine(x + w, y + h, x, y + h) 
                painter.drawLine(x, y + h, x, y)         

        # 2. Draw Live AI Predictions (Cyan/Blue with Text)
        if self.show_ai_preview and ai_predictions:
            painter.setRenderHint(QPainter.Antialiasing, True)
            pen_ai = QPen(QColor(0, 200, 255, 255))
            pen_ai.setWidth(2)
            painter.setPen(pen_ai)
            
            font = QFont("Arial", 10, QFont.Bold)
            painter.setFont(font)

            for cls_id, label, conf, x1, y1, x2, y2 in ai_predictions:
                painter.drawRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                painter.drawText(int(x1), int(y1) - 5, f"{label} {conf:.2f}")

        painter.end()

    # =========================================================================
    # EVENT CAPTURE AND ROUTING
    # =========================================================================

    def mousePressEvent(self, e: QMouseEvent) -> None:
        self.setFocus(Qt.MouseFocusReason)
        self.makeCurrent()
        self._controller.process_press(e.position().x(), e.position().y(), e.button(), self.width(), self.height())
        self.doneCurrent()

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        self.makeCurrent()
        self._controller.process_release(e.button())
        self.doneCurrent()

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        self.makeCurrent()
        self._controller.process_move(e.position().x(), e.position().y(), e.buttons(), self.width(), self.height())
        self.doneCurrent()

    def keyPressEvent(self, e: QKeyEvent) -> None:
        if e.isAutoRepeat():
            return

        ctrl = bool(e.modifiers() & Qt.ControlModifier)

        if ctrl and e.key() == Qt.Key_C:
            self.window().action_copy()
        elif ctrl and e.key() == Qt.Key_X:
            self.window().action_cut()
        elif ctrl and e.key() == Qt.Key_V:
            self.window().action_paste()
        elif e.key() in [Qt.Key_Delete, Qt.Key_Backspace]:
            self.window().action_delete()
        elif e.key() == Qt.Key_F and not ctrl:
            current = self.window().chk_wire.isChecked()
            self.window().chk_wire.setChecked(not current)
            
        elif e.key() == Qt.Key_B and not ctrl:
            # Toggle Mathematical Ground Truth
            self.show_debug_bboxes = not self.show_debug_bboxes
            self.update()
            
        elif e.key() == Qt.Key_I and not ctrl:
            # Toggle Live AI Inference
            if self.live_ai_engine is None:
                from src.ai_bridge.live_inference import LiveAIEngine
                # Fallback to YOLOv8 Nano base model if a custom best.pt is not specified yet
                self.live_ai_engine = LiveAIEngine("yolov8n.pt") 
                
            self.show_ai_preview = self.live_ai_engine.toggle()
            self.update()
            
        else:
            self._controller.process_key_press(e.key())

    def keyReleaseEvent(self, e: QKeyEvent) -> None:
        if e.isAutoRepeat():
            return
        self._controller.process_key_release(e.key())

    def wheelEvent(self, e: QWheelEvent) -> None:
        ctx.engine.zoom_camera(e.angleDelta().y() / 240.0)
        ctx.events.emit(AppEvent.SCENE_CHANGED)
        ctx.events.emit(AppEvent.ENTITY_SELECTED, ctx.engine.get_selected_entity_id())

    def focusOutEvent(self, e: QFocusEvent) -> None:
        self._controller.clear_keys()
        super().focusOutEvent(e)

    # =========================================================================
    # DRAG & DROP
    # =========================================================================

    def dragEnterEvent(self, e: QDragEnterEvent) -> None:
        if e.mimeData().hasText() and (e.mimeData().text().startswith("MODEL|") or e.mimeData().text().startswith("TEXTURE|")):
            e.setDropAction(Qt.CopyAction)
            e.accept()

    def dragMoveEvent(self, e: QDragMoveEvent) -> None:
        if e.mimeData().hasText() and (e.mimeData().text().startswith("MODEL|") or e.mimeData().text().startswith("TEXTURE|")):
            e.setDropAction(Qt.CopyAction)
            e.accept()

    def dropEvent(self, e: QDropEvent) -> None:
        parts = e.mimeData().text().split("|", 1)
        if len(parts) < 2:
            return

        asset_type, path = parts[0], parts[1]

        if asset_type == "MODEL":
            self.window()._controller.asset_ctrl.spawn_model(path)
        elif asset_type == "TEXTURE":
            x, y = int(e.position().x()), int(e.position().y())
            self.makeCurrent()

            hit_idx = ctx.engine.raycast_select(x, y, self.width(), self.height())
            self.doneCurrent()

            if hit_idx >= 0:
                ctx.engine.select_entity(hit_idx)
                ctx.events.emit(AppEvent.ENTITY_SELECTED, hit_idx)
                self._show_texture_mapping_menu(path)
            else:
                QMessageBox.warning(self, "Warning", "Dropped texture into empty space.\nPlease drop directly onto a 3D model.")

        e.setDropAction(Qt.CopyAction)
        e.accept()

    def _show_texture_mapping_menu(self, path: str) -> None:
        menu = QMenu(self)
        menu.setTitle("Apply Texture As:")
        menu.setStyleSheet(CONTEXT_MENU_STYLE)

        for label, attr_name in TEXTURE_CHANNELS.items():
            action = menu.addAction(f"Set as {label}")
            action.setData(attr_name)

        action_selected = menu.exec(QCursor.pos())
        if not action_selected:
            return

        map_attr = action_selected.data()
        self.window()._controller.asset_ctrl.apply_texture(path, map_attr)