"""
Viewport Interaction Controller.

Translates 2D mouse and keyboard inputs over the QOpenGLWidget into 
3D interactions such as Raycasting (Object Selection), Free-look Camera Navigation, 
and Gizmo Manipulation (Translate/Rotate/Scale).
"""

import time
from typing import Optional, Set, Tuple
from PySide6.QtCore import Qt

from src.ui.views.viewport.main_viewport import MainViewportView
from src.app import ctx, AppEvent, config
from src.ui.error_handler import safe_execute


class ViewportController:
    """Manages physical user interactions within the active 3D space."""
    
    def __init__(self, is_hud: bool = False) -> None:
        self.is_hud = is_hud
        
        if not self.is_hud:
            self.view = MainViewportView(controller=self)
        
        self.last_pos: Optional[Tuple[int, int]] = None
        self.active_axis: Optional[str] = None
        self.hovered_axis: Optional[str] = None
        self.hovered_screen_axis: Optional[str] = None
        
        self.last_time: float = time.perf_counter()
        self._drag_history_recorded: bool = False

        self.keymap = {
            Qt.Key_W: "CAM_FORWARD", Qt.Key_S: "CAM_BACKWARD",
            Qt.Key_A: "CAM_LEFT",    Qt.Key_D: "CAM_RIGHT",
            Qt.Key_Q: "CAM_ROLL_LEFT", Qt.Key_E: "CAM_ROLL_RIGHT"
        }
        self.active_commands: Set[str] = set()

    @safe_execute(context="Viewport Click Interaction")
    def process_press(self, x: int, y: int, button: Qt.MouseButton, width: int, height: int) -> None:
        """Evaluates intersections between the mouse cursor and 3D scene elements upon clicking."""
        self.last_pos = (x, y)
        self._drag_history_recorded = False
        
        if button == Qt.LeftButton:
            if self.is_hud:
                self.active_axis = ctx.engine.check_hud_gizmo_hover(x, y, width, height)
            else:
                # Snap camera to orthogonal view if compass axis is clicked
                if self.hovered_screen_axis:
                    ctx.events.emit(AppEvent.ACTION_BEFORE_MUTATION) 
                    ctx.engine.snap_camera_to_axis(self.hovered_screen_axis)
                    ctx.events.emit(AppEvent.SCENE_CHANGED)
                    return
                    
                # Lock onto transformation gizmo axis
                if self.hovered_axis:
                    self.active_axis = self.hovered_axis
                    return
                    
                # Standard Raycast object selection
                hit_idx = ctx.engine.raycast_select(x, y, width, height)
                ctx.engine.select_entity(hit_idx if hit_idx >= 0 else -1)
                ctx.events.emit(AppEvent.ENTITY_SELECTED, hit_idx)

    @safe_execute(context="Viewport Drag Interaction")
    def process_move(self, x: int, y: int, buttons: Qt.MouseButtons, width: int, height: int) -> None:
        """Processes continuous mouse dragging for orbit, panning, and gizmo manipulation."""
        if self.last_pos is None:
            self.last_pos = (x, y)
            dx = dy = 0
        else:
            dx = x - self.last_pos[0]
            dy = self.last_pos[1] - y
            self.last_pos = (x, y)

        # Hover states (No mouse buttons pressed)
        if buttons == Qt.NoButton:
            self.active_axis = None
            if self.is_hud:
                self.active_axis = ctx.engine.check_hud_gizmo_hover(x, y, width, height)
            else:
                self.hovered_screen_axis = ctx.engine.check_screen_axis_hover(x, y, width, height)
                if not self.hovered_screen_axis:
                    self.hovered_axis = ctx.engine.check_gizmo_hover(x, y, width, height)
                else:
                    self.hovered_axis = None
            
            if not self.is_hud:
                ctx.events.emit(AppEvent.SCENE_CHANGED)
            return

        should_redraw = False
        
        # Camera Orbit (Right Click)
        if buttons & Qt.RightButton and not self.is_hud:
            ctx.engine.orbit_camera(-dx, dy)
            should_redraw = True
            
        # Camera Pan (Middle Click)
        elif buttons & Qt.MiddleButton and not self.is_hud:
            ctx.engine.pan_camera(dx, -dy)
            should_redraw = True
            
        # Gizmo Manipulation (Left Click)
        elif buttons & Qt.LeftButton:
            if not self._drag_history_recorded:
                ctx.events.emit(AppEvent.ACTION_BEFORE_MUTATION)
                self._drag_history_recorded = True
                
            if self.is_hud:
                ctx.engine.handle_hud_gizmo_drag(dx, dy, self.active_axis, width, height)
                data = ctx.engine.get_selected_entity_data()
                if data and data.get("tf"):
                    rot = data["tf"].get("rotation", [0.0, 0.0, 0.0])
                    ctx.events.emit(AppEvent.TRANSFORM_FAST_UPDATED, ("ROTATE", (rot[0], rot[1], rot[2])))
            else:
                ctx.engine.handle_gizmo_drag(dx, dy, self.active_axis, width, height)
                tf_data = ctx.engine.get_selected_transform_state()
                if tf_data:
                    ctx.events.emit(AppEvent.TRANSFORM_FAST_UPDATED, tf_data)
                    
            should_redraw = True
            
        if should_redraw:
            ctx.events.emit(AppEvent.SCENE_CHANGED)

    def process_release(self, button: Qt.MouseButton) -> None:
        """Clears drag states and solidifies structural changes."""
        self._drag_history_recorded = False
        if button == Qt.LeftButton:
            self.active_axis = None
            curr_id = ctx.engine.get_selected_entity_id()
            ctx.events.emit(AppEvent.ENTITY_SELECTED, curr_id)

    def process_key_press(self, key: Qt.Key) -> None: 
        """Registers a WASD/QE movement command."""
        if key in self.keymap: 
            self.active_commands.add(self.keymap[key])
        
    def process_key_release(self, key: Qt.Key) -> None: 
        """Releases a movement command upon key lift."""
        if key in self.keymap and self.keymap[key] in self.active_commands: 
            self.active_commands.remove(self.keymap[key])
        
    def clear_keys(self) -> None: 
        """Flushes all queued key inputs (e.g., when the viewport loses focus)."""
        self.active_commands.clear()
        
    def process_continuous_input(self) -> bool:
        """
        Periodically polls the active keys to glide the camera smoothly.
        Monitors high frame variance to prevent camera teleportation artifacts.
        """
        if self.is_hud: 
            return False
            
        current_time = time.perf_counter()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        # Prevent physics/movement tunneling by locking maximum delta time during lag spikes
        if dt > 0.1: 
            dt = 1.0 / config.TARGET_FPS
            
        if not self.active_commands: 
            return False
            
        if ctx.engine.update_camera_movement(list(self.active_commands), dt):
            return True
            
        return False