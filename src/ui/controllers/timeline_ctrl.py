"""
Timeline Controller.

Coordinates the animation timeline, playhead state, and Dope Sheet UI.
Interfaces with the Engine's AnimatorSystem via Qt QTimers to provide 
real-time continuous playback capabilities.
"""

from typing import Any, Dict, Optional
from PySide6.QtCore import QTimer
import time

from src.app import ctx, AppEvent
from src.ui.error_handler import safe_execute
from src.ui.views.panels.timeline_view import TimelinePanelView


class TimelineController:
    """Manages playback state and keyframe manipulation logic."""
    
    def __init__(self) -> None:
        self.view = TimelinePanelView(controller=self)
        self.is_updating_ui: bool = False
        self.current_time: float = 0.0
        self.is_playing: bool = False
        self.generator_ctrl: Optional[Any] = None
        
        self.selected_kf_idx: int = -1
        self.current_entity_id: int = -1

        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self._on_playback_tick)
        self._last_time = 0.0

        ctx.events.subscribe(AppEvent.ENTITY_SELECTED, self._on_entity_selected)
        ctx.events.subscribe(AppEvent.SCENE_CHANGED, self._refresh_dope_sheet)

    @safe_execute(context="Select Keyframe")
    def select_keyframe(self, index: int) -> None:
        """Sets the active UI focus to a specific keyframe in the Dope Sheet."""
        self.selected_kf_idx = index
        target_time = ctx.engine.set_active_keyframe(index)
        
        if index >= 0:
            self.set_time(target_time)
            
        ctx.events.emit(AppEvent.COMPONENT_PROPERTY_CHANGED) 

    @safe_execute(context="Toggle Playback")
    def toggle_playback(self, is_playing: bool) -> None:
        """Starts or pauses the animation playback loop."""
        self.is_playing = is_playing
        if is_playing:
            self.select_keyframe(-1)
            self.view.deselect_keyframe_ui()
            self._last_time = time.time()
            self.playback_timer.start(16) # Roughly ~60fps target
        else:
            self.playback_timer.stop()

    def _on_playback_tick(self) -> None:
        """Triggered periodically by the QTimer to step the timeline forward."""
        now = time.time()
        dt = now - self._last_time
        self._last_time = now
        self.advance_time(dt)

    @safe_execute(context="Rewind Timeline")
    def rewind_timeline(self) -> None:
        """Returns the playhead to timestamp 0.0."""
        self.select_keyframe(-1)
        self.view.deselect_keyframe_ui()
        self.set_time(0.0)

    @safe_execute(context="Set Timeline Time")
    def set_time(self, time_sec: float) -> None:
        """Forces the scene state to reflect the interpolated values at a specific timestamp."""
        if self.selected_kf_idx != -1 and not self.is_playing:
            info = ctx.engine.get_animation_info()
            if info and info.get("active_idx", -1) == self.selected_kf_idx:
                time_sec = info.get("target_time", time_sec)

        self.current_time = time_sec
        self.view.update_ui_time(self.current_time)
        
        if hasattr(ctx.engine, 'animator'):
            ctx.engine.animator.evaluate(self.current_time, 0.0, target_entity_id=self.current_entity_id)
            if not self.is_playing:
                ctx.events.emit(AppEvent.COMPONENT_PROPERTY_CHANGED)
            ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Add Keyframe")
    def add_keyframe_at_current(self) -> None:
        """Bakes current spatial values into a new keyframe at the playhead position."""
        self.add_keyframe_at_time(self.current_time)

    @safe_execute(context="Add Keyframe At Time")
    def add_keyframe_at_time(self, time_val: float) -> None:
        """Bakes spatial values into a specific timestamp."""
        ctx.events.emit(AppEvent.ACTION_BEFORE_MUTATION)
        
        new_idx = ctx.engine.add_and_focus_keyframe(time_val)
        self._refresh_dope_sheet()
        
        if new_idx >= 0:
            self.selected_kf_idx = new_idx
            self.view.track.selected_indices = {new_idx}
            self.view.track.update()
            
        ctx.events.emit(AppEvent.COMPONENT_PROPERTY_CHANGED)
        ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Move Keyframe")
    def move_keyframe(self, index: int, new_time: float) -> None:
        """Shifts an existing keyframe horizontally along the timeline."""
        ctx.events.emit(AppEvent.ACTION_BEFORE_MUTATION)
        ctx.engine.set_component_property("Animation", "MOVE_KEYFRAME", {"index": index, "time": new_time})
        self._refresh_dope_sheet()
        
        self.current_time = new_time
        self.view.update_ui_time(self.current_time)
        
        if hasattr(ctx.engine, 'animator'):
            ctx.engine.animator.evaluate(self.current_time, 0.0, target_entity_id=self.current_entity_id)
            
        ctx.events.emit(AppEvent.COMPONENT_PROPERTY_CHANGED)
        ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Mutate Keyframes")
    def mutate_keyframes(self, payload: Dict[str, Any]) -> None:
        """Applies bulk modification commands to multiple keyframes simultaneously."""
        ctx.events.emit(AppEvent.ACTION_BEFORE_MUTATION)
        ctx.engine.set_component_property("Animation", "MUTATE_KEYFRAMES", payload)
        self._refresh_dope_sheet()
        
        if hasattr(ctx.engine, 'animator'):
            ctx.engine.animator.evaluate(self.current_time, 0.0, target_entity_id=self.current_entity_id)
            
        ctx.events.emit(AppEvent.COMPONENT_PROPERTY_CHANGED)
        ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Clear Keyframes")
    def clear_keyframes(self) -> None:
        """Deletes all animation tracks for the currently selected entity."""
        ctx.events.emit(AppEvent.ACTION_BEFORE_MUTATION)
        ctx.engine.set_component_property("Animation", "CLEAR_KEYFRAMES", None)
        self.selected_kf_idx = -1
        self._refresh_dope_sheet()
        
        if hasattr(ctx.engine, 'animator'):
            ctx.engine.animator.evaluate(self.current_time, 0.0, target_entity_id=self.current_entity_id)
            ctx.events.emit(AppEvent.COMPONENT_PROPERTY_CHANGED)
            
        ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Open Render Settings")
    def open_render_settings(self) -> None:
        """Launches the Synthetic Data Generation configuration dialog."""
        if self.generator_ctrl is None:
            from src.ui.controllers.generator_ctrl import GeneratorController
            self.generator_ctrl = GeneratorController()
        self.generator_ctrl.show_dialog()

    def advance_time(self, dt: float) -> None:
        """Steps the logical animation time forward by delta seconds."""
        if not self.is_playing:
            return
            
        self.current_time += dt
        max_time = self.view.spn_max_time.value()
        
        if self.current_time > max_time:
            self.current_time = 0.0 
            
        self.view.update_ui_time(self.current_time)
        
        if hasattr(ctx.engine, 'animator'):
            ctx.engine.animator.evaluate(self.current_time, dt, target_entity_id=self.current_entity_id)
            ctx.events.emit(AppEvent.SCENE_CHANGED)

    def _on_entity_selected(self, entity_id: int) -> None:
        """Callback responding to user selecting a different object in the scene."""
        if self.current_entity_id == entity_id:
            return 
            
        self.current_entity_id = entity_id

        if self.is_playing:
            self.select_keyframe(-1)
        else:
            if entity_id >= 0:
                self.select_keyframe(0)
            else:
                self.select_keyframe(-1)
                self.set_time(0.0)
            
        self._refresh_dope_sheet()
        
        if hasattr(ctx.engine, 'animator') and self.is_playing and not self.is_updating_ui:
            ctx.engine.animator.evaluate(self.current_time, 0.0, target_entity_id=self.current_entity_id)
            ctx.events.emit(AppEvent.COMPONENT_PROPERTY_CHANGED)
        
    def _refresh_dope_sheet(self) -> None:
        """Fetches active timeline data and redraws the graphical UI track markers."""
        info = ctx.engine.get_animation_info()
        
        if not info:
            self.view.update_keyframes_display([], "")
            return
            
        self.view.update_keyframes_display(info.get("times", []), "")
        
        active_idx = info.get("active_idx", -1)
        if self.selected_kf_idx != active_idx:
            self.selected_kf_idx = active_idx
            self.view.track.selected_indices = {self.selected_kf_idx} if self.selected_kf_idx >= 0 else set()
            self.view.track.update()