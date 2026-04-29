"""
Sun HUD Widget.

Provides a miniature 3D OpenGL viewport dedicated to controlling the 
orientation of the Directional Light in the scene.
"""

from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QMouseEvent
from typing import Optional

from src.app import ctx, AppEvent
from src.ui.controllers.viewport_ctrl import ViewportController
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
        
        self._controller = ViewportController(is_hud=True)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        hud_interval = max(1, int(1000 / max(TARGET_FPS, 1)))
        self.timer.start(hud_interval)

    def initializeGL(self) -> None:
        """Bootstraps the HUD OpenGL context via the Engine."""
        ctx.engine.init_hud_gl()

    def paintGL(self) -> None:
        """Renders the interactive compass interface to control the light direction."""
        ctx.engine.render_sun_hud(
            self.width(), 
            self.height(), 
            self._controller.active_axis, 
            self.underMouse()
        )

    def mousePressEvent(self, e: QMouseEvent) -> None: 
        """Delegates mouse press interactions (axis selection) to the HUD controller."""
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
        """Releases the dragging state of the selected axis."""
        self.makeCurrent()
        self._controller.process_release(e.button())
        self.doneCurrent()
        
    def mouseMoveEvent(self, e: QMouseEvent) -> None: 
        """Delegates mouse drag motions and syncs the rotation changes to the timeline."""
        self.makeCurrent()
        self._controller.process_move(
            e.position().x(), 
            e.position().y(), 
            e.buttons(), 
            self.width(), 
            self.height()
        )
        
        if e.buttons() & Qt.LeftButton:
            timeline = getattr(ctx.main_window._controller, 'timeline_ctrl', None) if hasattr(ctx, 'main_window') else None
            curr_time = timeline.current_time if timeline else 0.0
            
            is_new_kf, target_time = ctx.engine.sync_gizmo_to_keyframe(curr_time)
            
            if timeline:
                if is_new_kf:
                    timeline._refresh_dope_sheet()
                if abs(timeline.current_time - target_time) > 0.001:
                    timeline.set_time(target_time)
                elif hasattr(ctx.engine, 'animator'):
                    ctx.engine.animator.evaluate(target_time, 0.0)
            elif hasattr(ctx.engine, 'animator'):
                ctx.engine.animator.evaluate(target_time, 0.0)
                
            ctx.events.emit(AppEvent.SCENE_CHANGED)
            
            ctx.events.emit(AppEvent.COMPONENT_PROPERTY_CHANGED)
            
        self.doneCurrent()