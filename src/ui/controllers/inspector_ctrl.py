"""
Inspector Controller.
Coordinates the data flow for the Properties Panel, allowing real-time manipulation
of ECS components (Transform, Mesh, Material, Light, Camera).
"""

from typing import Any, Tuple
from src.app import ctx, AppEvent
from src.ui.error_handler import safe_execute
from src.ui.views.panels.inspector_view import InspectorPanelView

class InspectorController:
    """Coordinates data flow for the Properties (Inspector) panel."""
    
    def __init__(self) -> None:
        self.view = InspectorPanelView(controller=self)
        ctx.events.subscribe(AppEvent.ENTITY_SELECTED, self.on_entity_selected)
        ctx.events.subscribe(AppEvent.TRANSFORM_FAST_UPDATED, self.on_fast_transform_update)

    @safe_execute(context="Entity Selection")
    def on_entity_selected(self, entity_id: int) -> None:
        """Populates the Inspector UI with the active component state of the selected entity."""
        if entity_id < 0:
            self.view.hide_all_components()
            return
            
        data = ctx.engine.get_selected_entity_data()
        if data:
            self.view.update_inspector_data(data)
        else:
            self.view.hide_all_components()

    def on_fast_transform_update(self, transform_data: Tuple[str, Tuple[float, float, float]]) -> None:
        """Applies high-frequency transform updates driven by Viewport Gizmo dragging."""
        self.view.fast_update_transform(transform_data)

    def request_undo_snapshot(self) -> None:
        """Captures the current Scene state before applying discrete UI mutations."""
        ctx.events.emit(AppEvent.ACTION_BEFORE_MUTATION)

    @safe_execute(context="Modify Property")
    def set_property(self, comp_name: str, prop: str, value: Any) -> None:
        """Injects modified property values back into the underlying ECS component."""
        ctx.engine.set_component_property(comp_name, prop, value)
        ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Reset Transform")
    def reset_transform(self) -> None:
        """Resets the selected entity's spatial transform to the origin identity."""
        self.request_undo_snapshot()
        idx = ctx.engine.get_selected_entity_id()
        ctx.engine.reset_entity_transform(idx)
        ctx.events.emit(AppEvent.ENTITY_SELECTED, idx)
        ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Update Light Direction")
    def update_light_direction(self, yaw: float, pitch: float) -> None:
        """Applies explicit Yaw/Pitch spherical rotations to a directional/spot light source."""
        ctx.engine.update_light_direction(yaw, pitch)
        ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Load Texture")
    def load_texture(self, map_attr: str, filepath: str) -> None:
        """Assigns a discrete texture map slot on the selected entity's material."""
        self.request_undo_snapshot()
        
        if hasattr(ctx, 'main_window') and hasattr(ctx.main_window, 'gl_widget'):
            ctx.main_window.gl_widget.makeCurrent()
            ctx.engine.load_texture_to_selected(map_attr, filepath)
            ctx.main_window.gl_widget.doneCurrent()
        else:
            ctx.engine.load_texture_to_selected(map_attr, filepath)
            
        ctx.events.emit(AppEvent.ENTITY_SELECTED, ctx.engine.get_selected_entity_id())
        ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Remove Texture")
    def remove_texture(self, map_attr: str) -> None:
        """Unbinds a specific texture map slot from the selected entity's material."""
        self.request_undo_snapshot()
        ctx.engine.remove_texture_from_selected(map_attr)
        ctx.events.emit(AppEvent.ENTITY_SELECTED, ctx.engine.get_selected_entity_id())
        ctx.events.emit(AppEvent.SCENE_CHANGED)