from src.app import ctx, AppEvent
from src.ui.error_handler import safe_execute
from src.ui.views.panels.timeline_view import TimelinePanelView

class TimelineController:
    """
    Manages the global timeline, playback state, and synchronization between 
    the dope sheet UI and the animation engine.
    """
    def __init__(self) -> None:
        self.view = TimelinePanelView(controller=self)
        self.is_updating_ui: bool = False
        self.current_time: float = 0.0
        self.is_playing: bool = False
        self.generator_ctrl = None
        
        self.selected_kf_idx: int = -1
        self.current_entity_id: int = -1

        ctx.events.subscribe(AppEvent.ENTITY_SELECTED, self._on_entity_selected)
        ctx.events.subscribe(AppEvent.SCENE_CHANGED, self._refresh_dope_sheet)

    @safe_execute(context="Select Keyframe")
    def select_keyframe(self, index: int) -> None:
        self.selected_kf_idx = index
        target_time = ctx.engine.set_active_keyframe(index)
        
        if index >= 0 and abs(self.current_time - target_time) > 0.001:
            self.set_time(target_time)
            
        ctx.events.emit(AppEvent.COMPONENT_PROPERTY_CHANGED) 

    @safe_execute(context="Toggle Playback")
    def toggle_playback(self, is_playing: bool) -> None:
        self.is_playing = is_playing
        if is_playing:
            self.select_keyframe(-1)
            self.view.deselect_keyframe_ui()

    @safe_execute(context="Set Timeline Time")
    def set_time(self, time_sec: float) -> None:
        self.current_time = time_sec
        self.view.update_ui_time(self.current_time)
        
        if hasattr(ctx.engine, 'animator'):
            ctx.engine.animator.evaluate(self.current_time, 0.0)
            
            # Allow live visual updates of the inspector while scrubbing
            if not self.is_playing:
                ctx.events.emit(AppEvent.COMPONENT_PROPERTY_CHANGED)
                
            ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Add Keyframe")
    def add_keyframe_at_current(self) -> None:
        self.add_keyframe_at_time(self.current_time)

    @safe_execute(context="Add Keyframe At Time")
    def add_keyframe_at_time(self, time: float) -> None:
        ctx.events.emit(AppEvent.ACTION_BEFORE_MUTATION)
        
        new_idx = ctx.engine.add_and_focus_keyframe(time)
        self._refresh_dope_sheet()
        
        if new_idx >= 0:
            self.selected_kf_idx = new_idx
            self.view.track.selected_kf_index = new_idx
            self.view.track.update()
            
        ctx.events.emit(AppEvent.COMPONENT_PROPERTY_CHANGED)
        ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Move Keyframe")
    def move_keyframe(self, index: int, new_time: float) -> None:
        ctx.events.emit(AppEvent.ACTION_BEFORE_MUTATION)
        ctx.engine.set_component_property("Animation", "MOVE_KEYFRAME", {"index": index, "time": new_time})
        self._refresh_dope_sheet()
        
        if hasattr(ctx.engine, 'animator'):
            ctx.engine.animator.evaluate(self.current_time, 0.0)
            
        ctx.events.emit(AppEvent.COMPONENT_PROPERTY_CHANGED)
        ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Clear Keyframes")
    def clear_keyframes(self) -> None:
        ctx.events.emit(AppEvent.ACTION_BEFORE_MUTATION)
        ctx.engine.set_component_property("Animation", "CLEAR_KEYFRAMES", None)
        self._refresh_dope_sheet()
        
        if hasattr(ctx.engine, 'animator'):
            ctx.engine.animator.evaluate(self.current_time, 0.0)
            ctx.events.emit(AppEvent.COMPONENT_PROPERTY_CHANGED)
            
        ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Open Render Settings")
    def open_render_settings(self) -> None:
        if self.generator_ctrl is None:
            from src.ui.controllers.generator_ctrl import GeneratorController
            self.generator_ctrl = GeneratorController()
        self.generator_ctrl.show_dialog()

    def advance_time(self, dt: float) -> None:
        if not self.is_playing:
            info = ctx.engine.get_animation_info()
            if info.get("active_idx", -1) >= 0:
                if hasattr(ctx.engine, 'animator'):
                    ctx.engine.animator.evaluate(self.current_time, 0.0)
            return
            
        self.current_time += dt
        
        max_time = self.view.spn_max_time.value()
        if self.current_time > max_time:
            self.current_time = 0.0 
            
        self.view.update_ui_time(self.current_time)
        
        if hasattr(ctx.engine, 'animator'):
            ctx.engine.animator.evaluate(self.current_time, dt)
            ctx.events.emit(AppEvent.SCENE_CHANGED)

    def _on_entity_selected(self, entity_id: int) -> None:
        # CRITICAL FIX: Prevent viewport click-and-drag from resetting the timeline
        if self.current_entity_id == entity_id:
            return 
            
        self.current_entity_id = entity_id

        # Preserve current timeline time when changing selection to avoid camera transform jumps.
        if self.is_playing:
            self.select_keyframe(-1)
        else:
            info = ctx.engine.get_animation_info()
            self.selected_kf_idx = info.get("active_idx", -1)
            
        self._refresh_dope_sheet()
        
        if hasattr(ctx.engine, 'animator') and self.is_playing and not self.is_updating_ui:
            ctx.engine.animator.evaluate(self.current_time, 0.0)
            ctx.events.emit(AppEvent.COMPONENT_PROPERTY_CHANGED)
        
    def _refresh_dope_sheet(self) -> None:
        info = ctx.engine.get_animation_info()
        
        if not info:
            self.view.update_keyframes_display([], "")
            return
            
        self.view.update_keyframes_display(info.get("times", []), "")
        
        if self.selected_kf_idx != info.get("active_idx", -1):
            self.selected_kf_idx = info.get("active_idx", -1)
            self.view.track.selected_kf_index = self.selected_kf_idx
            self.view.track.update()
