"""
Main Viewport.

Provides the primary QOpenGLWidget responsible for rendering the 3D scene.
Handles low-level input events (mouse, keyboard, drag-and-drop) and delegates 
them to the Viewport Controller.
"""

from typing import Any, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import (QCursor, QDragEnterEvent, QDragMoveEvent,
                           QDropEvent, QFocusEvent, QKeyEvent, QMouseEvent,
                           QWheelEvent)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QLabel, QMenu, QMessageBox

from src.app import AppEvent, ctx
from src.app.config import (CONTEXT_MENU_STYLE, DEFAULT_BG_COLOR,
                            TEXTURE_CHANNELS, VIEWPORT_HUD_STYLE)


class MainViewportView(QOpenGLWidget):
    """
    Main 3D viewport (Dumb View).
    Delegates rendering to the Engine and handles peripheral events.
    """

    def __init__(self, controller: Any, parent: Optional[Any] = None) -> None:
        super().__init__(parent)
        self._controller = controller

        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.bg_color = DEFAULT_BG_COLOR

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
        """Bootstraps the Engine's OpenGL components once the context is ready."""
        ctx.engine.init_viewport_gl()

    def resizeGL(self, w: int, h: int) -> None:
        """Handles widget resizing and updates the Engine's camera aspect ratio."""
        ctx.engine.resize_gl(w, h)

    def paintGL(self) -> None:
        """Issues the render command to the Engine for the current frame."""
        active_axis = self._controller.active_axis
        hovered_axis = self._controller.hovered_axis
        hovered_screen_axis = self._controller.hovered_screen_axis

        ctx.engine.render_viewport(
            self.width(), self.height(), self.bg_color,
            active_axis, hovered_axis, hovered_screen_axis
        )

        self._update_hud_labels(hovered_screen_axis)

    def _update_hud_labels(self, hovered_screen_axis: str) -> None:
        """Updates the position and styling of the HUD compass labels."""
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

    # =========================================================================
    # EVENT CAPTURE AND ROUTING
    # =========================================================================

    def mousePressEvent(self, e: QMouseEvent) -> None:
        """Delegates mouse press interactions (like picking) to the controller."""
        self.setFocus(Qt.MouseFocusReason)
        self.makeCurrent()
        self._controller.process_press(e.position().x(), e.position().y(), e.button(), self.width(), self.height())
        self.doneCurrent()

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        """Clears drag states in the controller upon mouse release."""
        self.makeCurrent()
        self._controller.process_release(e.button())
        self.doneCurrent()

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        """Delegates continuous mouse dragging to the controller."""
        self.makeCurrent()
        self._controller.process_move(e.position().x(), e.position().y(), e.buttons(), self.width(), self.height())
        self.doneCurrent()

    def keyPressEvent(self, e: QKeyEvent) -> None:
        """Handles viewport-specific keyboard shortcuts and delegates movement keys."""
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
        else:
            self._controller.process_key_press(e.key())

    def keyReleaseEvent(self, e: QKeyEvent) -> None:
        """Releases the movement state for keys in the controller."""
        if e.isAutoRepeat():
            return
        self._controller.process_key_release(e.key())

    def wheelEvent(self, e: QWheelEvent) -> None:
        """Delegates mouse wheel scroll events to adjust camera zoom."""
        ctx.engine.zoom_camera(e.angleDelta().y() / 240.0)
        ctx.events.emit(AppEvent.SCENE_CHANGED)
        ctx.events.emit(AppEvent.ENTITY_SELECTED, ctx.engine.get_selected_entity_id())

    def focusOutEvent(self, e: QFocusEvent) -> None:
        """Clears all key commands when the viewport loses focus."""
        self._controller.clear_keys()
        super().focusOutEvent(e)

    # =========================================================================
    # DRAG & DROP
    # =========================================================================

    def dragEnterEvent(self, e: QDragEnterEvent) -> None:
        """Accepts structural elements dropping from the Asset Browser."""
        if e.mimeData().hasText() and (e.mimeData().text().startswith("MODEL|") or e.mimeData().text().startswith("TEXTURE|")):
            e.setDropAction(Qt.CopyAction)
            e.accept()

    def dragMoveEvent(self, e: QDragMoveEvent) -> None:
        """Accepts mouse moves while dragging items over the viewport."""
        if e.mimeData().hasText() and (e.mimeData().text().startswith("MODEL|") or e.mimeData().text().startswith("TEXTURE|")):
            e.setDropAction(Qt.CopyAction)
            e.accept()

    def dropEvent(self, e: QDropEvent) -> None:
        """Processes instantiation of models or application of textures dropped into the 3D scene."""
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
        """Spawns a context menu to choose which material channel receives the dropped texture."""
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