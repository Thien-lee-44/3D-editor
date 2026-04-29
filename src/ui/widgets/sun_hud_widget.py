"""
Sun HUD Widget.
Provides a secondary, isolated 3D OpenGL viewport dedicated to manipulating 
directional light (Sun) vectors via an interactive arcball interface.
"""

from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QMouseEvent
from typing import Optional

from src.app import ctx
from src.ui.controllers.viewport_ctrl import ViewportController

# Import centralized HUD sizing
from src.app.config import SUN_HUD_MIN_HEIGHT, TARGET_FPS

class SunHUDWidget(QOpenGLWidget):
    """
    Miniature 3D viewport dedicated to controlling the Directional Light orientation.
    Operates completely independently of the main viewport, utilizing its own EventBus 
    and a dedicated ViewportController instantiated in 'HUD mode'.
    """
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        
        self.setMouseTracking(True)
        self.setMinimumHeight(SUN_HUD_MIN_HEIGHT) 
        
        # Instantiate an isolated Controller specifically to capture interactions within this widget
        self._controller = ViewportController(is_hud=True)
        
        # The HUD requires a constant framerate for continuous arcball redrawing 
        # (effectively unlocking the render loop from the main application state)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        hud_interval = max(1, int(1000 / max(TARGET_FPS, 1)))
        self.timer.start(hud_interval)

    # =========================================================================
    # OPENGL LIFECYCLE EVENTS (Communicates with the Engine via global ctx)
    # =========================================================================

    def initializeGL(self) -> None:
        """Initializes the specific hardware state machine needed for the HUD."""
        ctx.engine.init_hud_gl()

    def paintGL(self) -> None:
        """
        Executes the specialized HUD render pass. 
        Delegates to the engine to highlight the rotational axis currently being hovered over.
        """
        ctx.engine.render_sun_hud(
            self.width(), 
            self.height(), 
            self._controller.active_axis, 
            self.underMouse()
        )

    # =========================================================================
    # MOUSE EVENT HANDLING (Delegated to isolated Controller logic)
    # =========================================================================

    def mousePressEvent(self, e: QMouseEvent) -> None: 
        self.makeCurrent()
        self._controller.process_press(
            e.position().x(), 
            e.position().y(), 
            e.button(), 
            self.width(), 
            self.height()
        )
        self.doneCurrent()
        
    def mouseReleaseEvent(self, e: QMouseEvent) -> None: 
        self.makeCurrent()
        self._controller.process_release(e.button())
        self.doneCurrent()
        
    def mouseMoveEvent(self, e: QMouseEvent) -> None: 
        self.makeCurrent()
        self._controller.process_move(
            e.position().x(), 
            e.position().y(), 
            e.buttons(), 
            self.width(), 
            self.height()
        )
        self.doneCurrent()