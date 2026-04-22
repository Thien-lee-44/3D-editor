from typing import Any
from src.app import ctx, AppEvent
from src.ui.error_handler import safe_execute
from src.ui.views.panels.inspector_view import InspectorPanelView

class InspectorController:
    """
    Coordinates data flow for the Properties (Inspector) panel.
    Delegates complex logic (Auto-Keying, Index Shifting) to the Engine Facade,
    ensuring strict synchronization between UI, live components, and keyframes.
    """
    def __init__(self) -> None:
        self.view = InspectorPanelView(controller=self)
        self._is_updating_ui = False  # Critical guard to prevent infinite UI echo loops
        
        ctx.events.subscribe(AppEvent.ENTITY_SELECTED, self.on_entity_selected)
        ctx.events.subscribe(AppEvent.TRANSFORM_FAST_UPDATED, self.on_fast_transform_update)
        ctx.events.subscribe(AppEvent.COMPONENT_PROPERTY_CHANGED, self.refresh_inspector)

    @safe_execute(context="Entity Selection")
    def on_entity_selected(self, entity_id: int) -> None:
        self.refresh_inspector()

    @safe_execute(context="Refresh Inspector")
    def refresh_inspector(self, *args: Any) -> None:
        self._is_updating_ui = True
        try:
            entity_id = ctx.engine.get_selected_entity_id()
            if entity_id < 0:
                self.view.hide_all_components()
                return
                
            data = ctx.engine.get_selected_entity_data()
            if data:
                info = ctx.engine.get_animation_info()
                data["active_keyframe_index"] = info.get("active_idx", -1)
                data["active_keyframe_time"] = info.get("target_time", 0.0)
                self.view.update_inspector_data(data)
            else:
                self.view.hide_all_components()
        finally:
            self._is_updating_ui = False

    def on_fast_transform_update(self, transform_data: tuple) -> None:
        self._is_updating_ui = True
        try:
            self.view.fast_update_transform(transform_data)
        finally:
            self._is_updating_ui = False
            
        timeline = getattr(ctx.main_window._controller, 'timeline_ctrl', None) if hasattr(ctx, 'main_window') else None
        curr_time = timeline.current_time if timeline else 0.0
        
        is_new_kf, target_time = ctx.engine.sync_gizmo_to_keyframe(curr_time)
        
        if timeline:
            if is_new_kf:
                timeline._refresh_dope_sheet()
            if abs(timeline.current_time - target_time) > 0.001:
                timeline.set_time(target_time)
                
        self.refresh_inspector()

    def request_undo_snapshot(self) -> None:
        ctx.events.emit(AppEvent.ACTION_BEFORE_MUTATION)

    @safe_execute(context="Modify Property")
    def set_property(self, comp_name: str, prop: str, value: Any) -> None:
        if self._is_updating_ui: 
            return # Block programmatic value changes from triggering Auto-Keying
            
        self.request_undo_snapshot()
        
        timeline = getattr(ctx.main_window._controller, 'timeline_ctrl', None) if hasattr(ctx, 'main_window') else None
        curr_time = timeline.current_time if timeline else 0.0
        
        is_kf_mode, is_new_kf, target_time = ctx.engine.update_keyframe_property(curr_time, comp_name, prop, value)
        
        if not is_kf_mode:
            ctx.engine.set_component_property(comp_name, prop, value)
        else:
            if timeline:
                if is_new_kf:
                    timeline._refresh_dope_sheet()
                if abs(timeline.current_time - target_time) > 0.001:
                    timeline.set_time(target_time)
                elif hasattr(ctx.engine, 'animator'):
                    ctx.engine.animator.evaluate(target_time, 0.0)
                        
        ctx.events.emit(AppEvent.SCENE_CHANGED)
        
        if comp_name == "Animation" or is_new_kf:
            self.refresh_inspector()

    @safe_execute(context="Select Keyframe From Inspector")
    def select_keyframe_from_inspector(self, index: int) -> None:
        if hasattr(ctx, 'main_window') and hasattr(ctx.main_window, '_controller'):
            timeline = getattr(ctx.main_window._controller, 'timeline_ctrl', None)
            if timeline:
                timeline.view.track.selected_kf_index = index
                timeline.view.track.update()
                timeline.select_keyframe(index)

    @safe_execute(context="Add Keyframe")
    def add_keyframe(self, time: float) -> None:
        if self._is_updating_ui: return
        self.request_undo_snapshot()
        
        if hasattr(ctx, 'main_window') and hasattr(ctx.main_window, '_controller'):
            timeline = getattr(ctx.main_window._controller, 'timeline_ctrl', None)
            if timeline:
                timeline.add_keyframe_at_time(time)
                
        ctx.events.emit(AppEvent.SCENE_CHANGED)
        self.refresh_inspector()

    @safe_execute(context="Remove Keyframe")
    def remove_keyframe(self, index: int) -> None:
        if self._is_updating_ui: return
        self.request_undo_snapshot()
        ctx.engine.set_component_property("Animation", "REMOVE_KEYFRAME", index)
        ctx.events.emit(AppEvent.SCENE_CHANGED)
        self.refresh_inspector()

    @safe_execute(context="Set Keyframe Time")
    def set_active_keyframe_time(self, time: float) -> None:
        if self._is_updating_ui: return
        self.request_undo_snapshot()
        
        info = ctx.engine.get_animation_info()
        active_idx = info.get("active_idx", -1)
        
        if active_idx > 0:
            if hasattr(ctx, 'main_window') and hasattr(ctx.main_window, '_controller'):
                timeline = getattr(ctx.main_window._controller, 'timeline_ctrl', None)
                if timeline:
                    timeline.move_keyframe(active_idx, time)
                    timeline.set_time(time)
                    return

            ctx.engine.set_component_property("Animation", "MOVE_KEYFRAME", {"index": active_idx, "time": time})
            ctx.events.emit(AppEvent.COMPONENT_PROPERTY_CHANGED)
            ctx.events.emit(AppEvent.SCENE_CHANGED)

    def get_semantic_classes(self) -> dict:
        return ctx.engine.get_semantic_classes()

    @safe_execute(context="Add Semantic Class")
    def add_semantic_class(self, name: str) -> int:
        if self._is_updating_ui: return 0
        self.request_undo_snapshot()
        new_id = ctx.engine.add_semantic_class(name)
        ctx.events.emit(AppEvent.SCENE_CHANGED)
        return new_id

    @safe_execute(context="Update Semantic Class Color")
    def update_semantic_class_color(self, class_id: int, color: list) -> None:
        if self._is_updating_ui: return
        self.request_undo_snapshot()
        ctx.engine.update_semantic_class_color(class_id, color)
        ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Reset Transform")
    def reset_transform(self) -> None:
        if self._is_updating_ui: return
        self.request_undo_snapshot()
        idx = ctx.engine.get_selected_entity_id()
        ctx.engine.reset_entity_transform(idx)
        ctx.events.emit(AppEvent.SCENE_CHANGED)
        self.refresh_inspector()

    @safe_execute(context="Update Light Direction")
    def update_light_direction(self, yaw: float, pitch: float) -> None:
        if self._is_updating_ui: return
        ctx.engine.update_light_direction(yaw, pitch)
        
        timeline = getattr(ctx.main_window._controller, 'timeline_ctrl', None) if hasattr(ctx, 'main_window') else None
        curr_time = timeline.current_time if timeline else 0.0
        
        is_kf_y, is_new_y, t_y = ctx.engine.update_keyframe_property(curr_time, "Light", "yaw", yaw)
        is_kf_p, is_new_p, t_p = ctx.engine.update_keyframe_property(curr_time, "Light", "pitch", pitch)
        
        if timeline and (is_kf_y or is_kf_p):
            if is_new_y or is_new_p:
                timeline._refresh_dope_sheet()
            if abs(timeline.current_time - t_y) > 0.001:
                timeline.set_time(t_y)
            elif hasattr(ctx.engine, 'animator'):
                ctx.engine.animator.evaluate(t_y, 0.0)
                
        ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Load Texture")
    def load_texture(self, map_attr: str, filepath: str) -> None:
        if self._is_updating_ui: return
        self.request_undo_snapshot()
        timeline = getattr(ctx.main_window._controller, 'timeline_ctrl', None) if hasattr(ctx, 'main_window') else None
        curr_time = timeline.current_time if timeline else 0.0
        if curr_time > 0.01 and hasattr(ctx.engine, 'animator'):
            # Prime animation state before mutating material to preserve base snapshot.
            ctx.engine.animator.evaluate(curr_time, 0.0)
        if hasattr(ctx, 'main_window') and hasattr(ctx.main_window, 'gl_widget'):
            ctx.main_window.gl_widget.makeCurrent()
            ctx.engine.load_texture_to_selected(map_attr, filepath)
            ctx.main_window.gl_widget.doneCurrent()
        else:
            ctx.engine.load_texture_to_selected(map_attr, filepath)

        data = ctx.engine.get_selected_entity_data()
        
        if data and "mesh" in data and "mat_tex_paths" in data["mesh"]:
            is_kf, is_new, t_time = ctx.engine.update_keyframe_property(
                curr_time, "Mesh", "mat_tex_paths", data["mesh"]["mat_tex_paths"]
            )
            if timeline and is_kf:
                if is_new: 
                    timeline._refresh_dope_sheet()
                if abs(timeline.current_time - t_time) > 0.001: 
                    timeline.set_time(t_time)
                elif hasattr(ctx.engine, 'animator'):
                    ctx.engine.animator.evaluate(t_time, 0.0)
            
        ctx.events.emit(AppEvent.SCENE_CHANGED)
        self.refresh_inspector()

    @safe_execute(context="Remove Texture")
    def remove_texture(self, map_attr: str) -> None:
        if self._is_updating_ui: return
        self.request_undo_snapshot()
        timeline = getattr(ctx.main_window._controller, 'timeline_ctrl', None) if hasattr(ctx, 'main_window') else None
        curr_time = timeline.current_time if timeline else 0.0
        if curr_time > 0.01 and hasattr(ctx.engine, 'animator'):
            # Prime animation state before mutating material to preserve base snapshot.
            ctx.engine.animator.evaluate(curr_time, 0.0)
        ctx.engine.remove_texture_from_selected(map_attr)

        data = ctx.engine.get_selected_entity_data()
        
        if data and "mesh" in data and "mat_tex_paths" in data["mesh"]:
            is_kf, is_new, t_time = ctx.engine.update_keyframe_property(
                curr_time, "Mesh", "mat_tex_paths", data["mesh"]["mat_tex_paths"]
            )
            if timeline and is_kf:
                if is_new: 
                    timeline._refresh_dope_sheet()
                if abs(timeline.current_time - t_time) > 0.001: 
                    timeline.set_time(t_time)
                elif hasattr(ctx.engine, 'animator'):
                    ctx.engine.animator.evaluate(t_time, 0.0)
                
        ctx.events.emit(AppEvent.SCENE_CHANGED)
        self.refresh_inspector()
