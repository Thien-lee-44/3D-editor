"""
Inspector Controller.

Coordinates data flow for the Properties (Inspector) panel.
Maps user inputs directly into Component data structures and integrates 
tightly with the Animation Timeline.
"""

from typing import Any, Dict, Tuple
from src.app import ctx, AppEvent
from src.ui.error_handler import safe_execute
from src.ui.views.panels.inspector_view import InspectorPanelView


class InspectorController:
    """
    Coordinates data flow for the Properties (Inspector) panel.
    Utilizes Atomic Transactions (set_properties) to eliminate event storming 
    and maintain strict UI-Engine synchronization without latency.
    """

    def __init__(self) -> None:
        self.view = InspectorPanelView(controller=self)
        self._is_updating_ui = False  
        
        ctx.events.subscribe(AppEvent.ENTITY_SELECTED, self.on_entity_selected)
        ctx.events.subscribe(AppEvent.TRANSFORM_FAST_UPDATED, self.on_fast_transform_update)
        ctx.events.subscribe(AppEvent.COMPONENT_PROPERTY_CHANGED, self.refresh_inspector)

    @safe_execute(context="Entity Selection")
    def on_entity_selected(self, entity_id: int) -> None:
        """Refreshes the inspector context when a new entity is selected."""
        self.refresh_inspector()

    @safe_execute(context="Refresh Inspector")
    def refresh_inspector(self, *args: Any) -> None:
        """Pulls the latest component data from the Engine to populate the UI widgets."""
        self._is_updating_ui = True
        try:
            entity_id = ctx.engine.get_selected_entity_id()
            if entity_id < 0:
                self.view.hide_all_components()
                return
                
            data = ctx.engine.get_selected_entity_data()
            if data:
                # Merge animation and semantic tracking data into the UI payload
                info = ctx.engine.get_animation_info()
                data["active_keyframe_index"] = info.get("active_idx", -1)
                data["active_keyframe_time"] = info.get("target_time", 0.0)
                
                if data.get("semantic") and ctx.engine.scene and 0 <= entity_id < len(ctx.engine.scene.entities):
                    ent = ctx.engine.scene.entities[entity_id]
                    data["semantic"]["resolved_track_id"] = ctx.engine.get_resolved_track_id(ent)
                
                self.view.update_inspector_data(data)
            else:
                self.view.hide_all_components()
        finally:
            self._is_updating_ui = False

    def on_fast_transform_update(self, transform_data: Tuple[str, Tuple[float, float, float]]) -> None:
        """Bypasses full rebuilds to rapidly update SpinBoxes during Gizmo dragging."""
        self._is_updating_ui = True
        try:
            self.view.fast_update_transform(transform_data)
        finally:
            self._is_updating_ui = False
            
        timeline = getattr(ctx.main_window._controller, 'timeline_ctrl', None) if hasattr(ctx, 'main_window') else None
        curr_time = timeline.current_time if timeline else 0.0
        
        is_new_kf, target_time = ctx.engine.sync_gizmo_to_keyframe(curr_time, is_hud_drag=True)
        
        if timeline:
            if is_new_kf:
                timeline._refresh_dope_sheet()
            if abs(timeline.current_time - target_time) > 0.001:
                timeline.set_time(target_time)
                
        self.refresh_inspector()

    def request_undo_snapshot(self) -> None:
        """Triggers an Undo save state prior to a modification."""
        ctx.events.emit(AppEvent.ACTION_BEFORE_MUTATION)

    @safe_execute(context="Modify Property")
    def set_property(self, comp_name: str, prop: str, value: Any) -> None:
        """Legacy singular fallback for isolated UI components."""
        self.set_properties(comp_name, {prop: value})

    @safe_execute(context="Modify Properties Batch")
    def set_properties(self, comp_name: str, payload: Dict[str, Any]) -> None:
        """
        Executes an atomic transaction for multiple property assignments.
        Guarantees that 1 UI interaction creates exactly 1 render cycle and 1 keyframe snapshot.
        """
        if self._is_updating_ui: 
            return 
            
        self.request_undo_snapshot()
        
        timeline = getattr(ctx.main_window._controller, 'timeline_ctrl', None) if hasattr(ctx, 'main_window') else None
        curr_time = timeline.current_time if timeline else 0.0
        
        is_kf_mode, is_new_kf, target_time = ctx.engine.update_keyframe_properties(curr_time, comp_name, payload)
        
        if not is_kf_mode:
            ctx.engine.set_component_properties(comp_name, payload)
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

    @safe_execute(context="Select Keyframe")
    def select_keyframe_from_inspector(self, index: int) -> None:
        """Syncs the inspector selection state back to the Timeline Dope Sheet."""
        if hasattr(ctx, 'main_window') and hasattr(ctx.main_window, '_controller'):
            timeline = getattr(ctx.main_window._controller, 'timeline_ctrl', None)
            if timeline:
                timeline.view.track.selected_kf_index = index
                timeline.view.track.update()
                timeline.select_keyframe(index)

    @safe_execute(context="Add Keyframe")
    def add_keyframe(self, time: float) -> None:
        """Delegates keyframe addition to the Timeline controller."""
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
        """Deletes a specific keyframe track index."""
        if self._is_updating_ui: return
        self.request_undo_snapshot()
        ctx.engine.set_component_properties("Animation", {"REMOVE_KEYFRAME": index})
        ctx.events.emit(AppEvent.SCENE_CHANGED)
        self.refresh_inspector()

    @safe_execute(context="Set Keyframe Time")
    def set_active_keyframe_time(self, time: float) -> None:
        """Shifts an existing keyframe horizontally on the timeline from the Inspector input."""
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

            ctx.engine.set_component_properties("Animation", {"MOVE_KEYFRAME": {"index": active_idx, "time": time}})
            ctx.events.emit(AppEvent.COMPONENT_PROPERTY_CHANGED)
            ctx.events.emit(AppEvent.SCENE_CHANGED)

    def get_semantic_classes(self) -> dict:
        """Retrieves the list of AI dataset classes."""
        return ctx.engine.get_semantic_classes()

    @safe_execute(context="Add Semantic Class")
    def add_semantic_class(self, name: str) -> int:
        """Registers a new AI labeling class."""
        if self._is_updating_ui: return 0
        self.request_undo_snapshot()
        new_id = ctx.engine.add_semantic_class(name)
        ctx.events.emit(AppEvent.SCENE_CHANGED)
        return new_id

    @safe_execute(context="Update Semantic Class Color")
    def update_semantic_class_color(self, class_id: int, color: list) -> None:
        """Updates the visual UI color representing a specific class."""
        if self._is_updating_ui: return
        self.request_undo_snapshot()
        ctx.engine.update_semantic_class_color(class_id, color)
        ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Remove Semantic Class")
    def remove_semantic_class(self, class_id: int) -> None:
        """Deletes a custom semantic class from the registry."""
        if self._is_updating_ui: return
        self.request_undo_snapshot()
        ctx.engine.remove_semantic_class(class_id)
        ctx.events.emit(AppEvent.SCENE_CHANGED)
        self.refresh_inspector()

    @safe_execute(context="Reset Transform")
    def reset_transform(self) -> None:
        """Resets position, rotation, and scale back to defaults."""
        if self._is_updating_ui: return
        self.request_undo_snapshot()
        idx = ctx.engine.get_selected_entity_id()
        
        ctx.engine.reset_entity_transform(idx)
        
        timeline = getattr(ctx.main_window._controller, 'timeline_ctrl', None) if hasattr(ctx, 'main_window') else None
        curr_time = timeline.current_time if timeline else 0.0
        
        payload = {
            "position": [0.0, 0.0, 0.0],
            "rotation": [0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0]
        }
        ctx.engine.update_keyframe_properties(curr_time, "Transform", payload)
        
        ctx.events.emit(AppEvent.SCENE_CHANGED)
        self.refresh_inspector()

    @safe_execute(context="Update Light Direction")
    def update_light_direction(self, yaw: float, pitch: float) -> None:
        """Specifically handles updates to Directional/Spotlight Euler angles."""
        if self._is_updating_ui: return
        ctx.engine.update_light_direction(yaw, pitch)
        
        timeline = getattr(ctx.main_window._controller, 'timeline_ctrl', None) if hasattr(ctx, 'main_window') else None
        curr_time = timeline.current_time if timeline else 0.0
        
        payload = {"yaw": yaw, "pitch": pitch}
        is_kf, is_new, target_time = ctx.engine.update_keyframe_properties(curr_time, "Light", payload)
        
        if timeline and is_kf:
            if is_new:
                timeline._refresh_dope_sheet()
            if abs(timeline.current_time - target_time) > 0.001:
                timeline.set_time(target_time)
            elif hasattr(ctx.engine, 'animator'):
                ctx.engine.animator.evaluate(target_time, 0.0)
                
        ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Load Texture")
    def load_texture(self, map_attr: str, filepath: str) -> None:
        """Binds a texture file to a material slot directly from the Inspector dialog."""
        if self._is_updating_ui: return
        self.request_undo_snapshot()
        timeline = getattr(ctx.main_window._controller, 'timeline_ctrl', None) if hasattr(ctx, 'main_window') else None
        curr_time = timeline.current_time if timeline else 0.0
        
        if curr_time > 0.01 and hasattr(ctx.engine, 'animator'):
            ctx.engine.animator.evaluate(curr_time, 0.0)
            
        if hasattr(ctx, 'main_window') and hasattr(ctx.main_window, 'gl_widget'):
            ctx.main_window.gl_widget.makeCurrent()
            ctx.engine.load_texture_to_selected(map_attr, filepath)
            ctx.main_window.gl_widget.doneCurrent()
        else:
            ctx.engine.load_texture_to_selected(map_attr, filepath)

        data = ctx.engine.get_selected_entity_data()
        
        if data and "mesh" in data and "mat_tex_paths" in data["mesh"]:
            is_kf, is_new, t_time = ctx.engine.update_keyframe_properties(
                curr_time, "Mesh", {"mat_tex_paths": data["mesh"]["mat_tex_paths"]}
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
        """Unbinds a texture from the active material slot."""
        if self._is_updating_ui: return
        self.request_undo_snapshot()
        timeline = getattr(ctx.main_window._controller, 'timeline_ctrl', None) if hasattr(ctx, 'main_window') else None
        curr_time = timeline.current_time if timeline else 0.0
        
        if curr_time > 0.01 and hasattr(ctx.engine, 'animator'):
            ctx.engine.animator.evaluate(curr_time, 0.0)
            
        ctx.engine.remove_texture_from_selected(map_attr)

        data = ctx.engine.get_selected_entity_data()
        
        if data and "mesh" in data and "mat_tex_paths" in data["mesh"]:
            is_kf, is_new, t_time = ctx.engine.update_keyframe_properties(
                curr_time, "Mesh", {"mat_tex_paths": data["mesh"]["mat_tex_paths"]}
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