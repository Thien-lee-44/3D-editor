"""
Main Viewport View.
Serves as the primary 3D rendering canvas. Captures user inputs (Mouse, Keyboard, Drag & Drop)
and routes them to the Viewport Controller while executing OpenGL lifecycle events.
"""

from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import Qt
from PySide6.QtGui import (QMouseEvent, QKeyEvent, QWheelEvent, 
                           QDragEnterEvent, QDragMoveEvent, QDropEvent, QFocusEvent, QCursor)
from PySide6.QtWidgets import QLabel, QMessageBox, QMenu
from typing import Any, Optional, Dict

from src.app import ctx, AppEvent

# Import SSOT configuration
from src.app.config import TEXTURE_CHANNELS, DEFAULT_BG_COLOR, VIEWPORT_HUD_STYLE, CONTEXT_MENU_STYLE

class MainViewportView(QOpenGLWidget):
    """
    Main 3D viewport (Dumb View).
    Strictly responsible for executing OpenGL drawing functions and capturing 
    peripheral events. Logic is completely deferred to the associated Controller.
    """
    
    def __init__(self, controller: Any, parent: Optional[Any] = None) -> None:
        super().__init__(parent)
        self._controller = controller
        
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.bg_color: tuple = DEFAULT_BG_COLOR 
        
        # Labels for displaying X, Y, Z axes on the screen (HUD)
        self.lbl_x = QLabel("X", self)
        self.lbl_y = QLabel("Y", self)
        self.lbl_z = QLabel("Z", self)
        self.lbl_nx = QLabel("-X", self)
        self.lbl_ny = QLabel("-Y", self)
        self.lbl_nz = QLabel("-Z", self)
        
        self.labels_dict: Dict[str, QLabel] = {
            'X': self.lbl_x, 'Y': self.lbl_y, 'Z': self.lbl_z, 
            '-X': self.lbl_nx, '-Y': self.lbl_ny, '-Z': self.lbl_nz
        }

    # =========================================================================
    # OPENGL LIFECYCLE
    # =========================================================================

    def initializeGL(self) -> None:
        """Called once by Qt to set up the initial OpenGL state."""
        ctx.engine.init_viewport_gl()

    def resizeGL(self, w: int, h: int) -> None: 
        """Called by Qt whenever the widget is resized."""
        ctx.engine.resize_gl(w, h)

    def paintGL(self) -> None:
        """Retrieves data from the Engine to draw visual components to the screen."""
        active_axis = self._controller.active_axis
        hovered_axis = self._controller.hovered_axis
        hovered_screen_axis = self._controller.hovered_screen_axis

        ctx.engine.render_viewport(
            self.width(), self.height(), self.bg_color, 
            active_axis, hovered_axis, hovered_screen_axis
        )
        
        # Reset and reposition HUD orientation labels
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
        """Clears active input state to prevent stuck key interactions when losing focus."""
        self._controller.clear_keys()
        super().focusOutEvent(e)

    # =========================================================================
    # DRAG & DROP
    # =========================================================================
    
    def dragEnterEvent(self, e: QDragEnterEvent) -> None:
        """Validates incoming internal drag data and explicitly forces a Copy action."""
        if e.mimeData().hasText() and (e.mimeData().text().startswith("MODEL|") or e.mimeData().text().startswith("TEXTURE|")): 
            e.setDropAction(Qt.CopyAction)
            e.accept()

    def dragMoveEvent(self, e: QDragMoveEvent) -> None:
        """Maintains the 'Copy' action explicitly during mouse movement over the OpenGL canvas."""
        if e.mimeData().hasText() and (e.mimeData().text().startswith("MODEL|") or e.mimeData().text().startswith("TEXTURE|")): 
            e.setDropAction(Qt.CopyAction)
            e.accept()

    def dropEvent(self, e: QDropEvent) -> None:
        """Handles the event when the user drops an internal resource into the 3D viewport."""
        parts = e.mimeData().text().split("|", 1)
        if len(parts) < 2: 
            return
            
        asset_type, path = parts[0], parts[1]
        
        if asset_type == "MODEL": 
            # Call through parent hierarchy
            self.window()._controller.asset_ctrl.spawn_model(path)
            
        elif asset_type == "TEXTURE":
            x, y = int(e.position().x()), int(e.position().y())
            self.makeCurrent()
            
            # Request the Engine to raycast to see which object the mouse is over
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
        """Displays a dynamic context menu to select the Texture channel."""
        menu = QMenu(self)
        menu.setTitle("Apply Texture As:")
        menu.setStyleSheet(CONTEXT_MENU_STYLE)
        
        # Populate dynamic menu items directly from the central Config
        for label, attr_name in TEXTURE_CHANNELS.items():
            action = menu.addAction(f"Set as {label}")
            action.setData(attr_name) 
            
        action_selected = menu.exec(QCursor.pos())
        if not action_selected: 
            return
        
        map_attr = action_selected.data()
        self.window()._controller.asset_ctrl.apply_texture(path, map_attr)