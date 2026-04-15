from src.app import ctx, AppEvent
from src.ui.error_handler import safe_execute
from src.ui.views.panels.timeline_view import TimelinePanelView
from src.engine.scene.components.animation_cmp import AnimationComponent

class TimelineController:
    """
    Orchestrates the global playback state and synchronizes the UI timeline 
    with the underlying Engine's AnimatorSystem.
    """
    def __init__(self) -> None:
        self.view = TimelinePanelView(controller=self)
        
        self.is_updating_ui: bool = False
        self.current_time: float = 0.0
        self.is_playing: bool = False
        self.generator_ctrl = None

        ctx.events.subscribe(AppEvent.ENTITY_SELECTED, self._on_entity_selected)
        ctx.events.subscribe(AppEvent.SCENE_CHANGED, self._refresh_dope_sheet)

    @safe_execute(context="Select Keyframe")
    def select_keyframe(self, index: int) -> None:
        """Sets the active keyframe for direct parameter overriding."""
        ent_id = ctx.engine.get_selected_entity_id()
        if ent_id < 0: return
        
        ent = ctx.engine.scene.entities[ent_id]
        anim = ent.get_component(AnimationComponent)
        
        if anim:
            anim.active_keyframe_index = index
            if 0 <= index < len(anim.keyframes):
                target_time = anim.keyframes[index].time
                self.set_time(target_time)

    @safe_execute(context="Toggle Playback")
    def toggle_playback(self, is_playing: bool) -> None:
        self.is_playing = is_playing
        if is_playing:
            # Drop auto-keying focus for safety during playback
            self.select_keyframe(-1)
            self.view.deselect_keyframe_ui()

    @safe_execute(context="Set Timeline Time")
    def set_time(self, time_sec: float) -> None:
        self.current_time = time_sec
        self.view.update_ui_time(self.current_time)
        
        if hasattr(ctx.engine, 'animator'):
            ctx.engine.animator.evaluate(self.current_time, 0.0)
            ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Add Keyframe")
    def add_keyframe_at_current(self) -> None:
        self.add_keyframe_at_time(self.current_time)

    @safe_execute(context="Add Keyframe At Time")
    def add_keyframe_at_time(self, time: float) -> None:
        ctx.events.emit(AppEvent.ACTION_BEFORE_MUTATION)
        ctx.engine.set_component_property("Animation", "ADD_KEYFRAME", time)
        self._refresh_dope_sheet()
        ctx.events.emit(AppEvent.ENTITY_SELECTED, ctx.engine.get_selected_entity_id())
        ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Move Keyframe")
    def move_keyframe(self, index: int, new_time: float) -> None:
        ent_id = ctx.engine.get_selected_entity_id()
        if ent_id < 0: return
        
        ent = ctx.engine.scene.entities[ent_id]
        anim = ent.get_component(AnimationComponent)
        if not anim or index < 0 or index >= len(anim.keyframes): return
        
        ctx.events.emit(AppEvent.ACTION_BEFORE_MUTATION)
        
        kf_target = anim.keyframes[index]
        kf_target.time = new_time
        
        if hasattr(anim, '_sort_and_update_duration'):
            anim._sort_and_update_duration()
            
        self._refresh_dope_sheet()
        ctx.events.emit(AppEvent.ENTITY_SELECTED, ent_id)
        ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Clear Keyframes")
    def clear_keyframes(self) -> None:
        ctx.events.emit(AppEvent.ACTION_BEFORE_MUTATION)
        ctx.engine.set_component_property("Animation", "CLEAR_KEYFRAMES", None)
        
        ent_id = ctx.engine.get_selected_entity_id()
        if ent_id >= 0:
            ent = ctx.engine.scene.entities[ent_id]
            anim = ent.get_component(AnimationComponent)
            if anim:
                from src.engine.scene.components.animation_cmp import Keyframe
                anim.add_keyframe(Keyframe(0.0))
                
        self._refresh_dope_sheet()
        ctx.events.emit(AppEvent.ENTITY_SELECTED, ctx.engine.get_selected_entity_id())
        ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Open Render Settings")
    def open_render_settings(self) -> None:
        """Instantiates and presents the floating Data Generator dialog."""
        if self.generator_ctrl is None:
            from src.ui.controllers.generator_ctrl import GeneratorController
            self.generator_ctrl = GeneratorController()
        self.generator_ctrl.show_dialog()

    def advance_time(self, dt: float) -> None:
        if not self.is_playing:
            # =================================================================
            # [CRITICAL FIX]: VIEWPORT AUTO-KEYING SYNCHRONIZATION
            # Even when playback is paused, if an entity is in Edit Mode 
            # (a keyframe is selected), we must explicitly wake up the Animator 
            # so it can poll and bake real-time Viewport Gizmo manipulations.
            # =================================================================
            ent_id = ctx.engine.get_selected_entity_id()
            if ent_id >= 0:
                ent = ctx.engine.scene.entities[ent_id]
                anim = ent.get_component(AnimationComponent)
                if anim and hasattr(anim, 'active_keyframe_index') and anim.active_keyframe_index >= 0:
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
        self._refresh_dope_sheet()
        
    def _refresh_dope_sheet(self) -> None:
        ent_id = ctx.engine.get_selected_entity_id()
        if ent_id < 0:
            self.view.update_keyframes_display([], "")
            return
            
        ent = ctx.engine.scene.entities[ent_id]
        anim = ent.get_component(AnimationComponent)
        
        if anim and anim.keyframes:
            times = [k.time for k in anim.keyframes]
            self.view.update_keyframes_display(times, ent.name)
        else:
            self.view.update_keyframes_display([], ent.name)