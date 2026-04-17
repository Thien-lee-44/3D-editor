from PySide6.QtCore import QPoint
from typing import List
from src.app import ctx, AppEvent
from src.ui.error_handler import safe_execute
from src.ui.views.panels.hierarchy_view import HierarchyPanelView

class HierarchyController:
    """Manages the logic and state of the Scene Graph Hierarchy tree."""
    def __init__(self) -> None:
        self.view = HierarchyPanelView(controller=self)
        
        self.selected_multi_ids: List[int] = [] 
        
        ctx.events.subscribe(AppEvent.HIERARCHY_NEEDS_REFRESH, self.refresh_view)
        ctx.events.subscribe(AppEvent.ENTITY_SELECTED, self.on_global_selection)

    @safe_execute(context="Refresh Hierarchy")
    def refresh_view(self) -> None:
        entities_data = ctx.engine.get_scene_entities_list()
        selected_idx = ctx.engine.get_selected_entity_id()
        self.view.build_tree(entities_data, selected_idx)

    def on_global_selection(self, entity_id: int) -> None:
        self.view.update_selection(entity_id)

    @safe_execute(context="Select Entity")
    def handle_item_selected(self, entity_id: int) -> None:
        ctx.engine.select_entity(entity_id)
        ctx.events.emit(AppEvent.ENTITY_SELECTED, entity_id)
        ctx.events.emit(AppEvent.SCENE_CHANGED)

    def handle_multi_selection(self, ids: List[int]) -> None:
        """Caches the list of currently highlighted items for group/ungroup operations."""
        self.selected_multi_ids = ids

    @safe_execute(context="Reorder Hierarchy")
    def handle_hierarchy_reorder(self, hierarchy_mapping: dict) -> None:
        ctx.engine.sync_hierarchy_from_ui(hierarchy_mapping)
        ctx.events.emit(AppEvent.HIERARCHY_NEEDS_REFRESH)
        ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Rename Entity")
    def handle_rename(self, entity_id: int, new_name: str) -> None:
        """Synchronizes an inline hierarchy rename directly into the engine state."""
        ctx.events.emit(AppEvent.ACTION_BEFORE_MUTATION)
        
        if hasattr(ctx.engine, 'rename_entity'):
            ctx.engine.rename_entity(entity_id, new_name)
        else:
            # Fallback logic leveraging existing API if a direct rename method is unavailable
            prev_id = ctx.engine.get_selected_entity_id()
            ctx.engine.select_entity(entity_id)
            ctx.engine.set_component_property("Entity", "name", new_name)
            ctx.engine.select_entity(prev_id)
            
        if ctx.engine.get_selected_entity_id() == entity_id:
            ctx.events.emit(AppEvent.COMPONENT_PROPERTY_CHANGED)
            
        ctx.events.emit(AppEvent.SCENE_CHANGED)

    def show_context_menu(self, global_pos: QPoint) -> None:
        if hasattr(ctx, 'main_window') and hasattr(ctx.main_window, 'show_context_menu'):
            ctx.main_window.show_context_menu(global_pos)

    @safe_execute(context="Copy Entity")
    def handle_copy(self) -> None:
        ctx.engine.copy_selected()

    @safe_execute(context="Cut Entity")
    def handle_cut(self) -> None:
        ctx.events.emit(AppEvent.ACTION_BEFORE_MUTATION) 
        ctx.engine.cut_selected()
        ctx.events.emit(AppEvent.HIERARCHY_NEEDS_REFRESH)
        ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Paste Entity")
    def handle_paste(self) -> None:
        ctx.events.emit(AppEvent.ACTION_BEFORE_MUTATION)
        if hasattr(ctx, 'main_window') and hasattr(ctx.main_window, 'gl_widget'):
            ctx.main_window.gl_widget.makeCurrent()
            ctx.engine.paste_copied()
            ctx.main_window.gl_widget.doneCurrent()
        else:
            ctx.engine.paste_copied()
            
        ctx.events.emit(AppEvent.HIERARCHY_NEEDS_REFRESH)
        ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Delete Entity")
    def handle_delete(self) -> None:
        ctx.events.emit(AppEvent.ACTION_BEFORE_MUTATION)
        ctx.engine.delete_selected()
        ctx.events.emit(AppEvent.HIERARCHY_NEEDS_REFRESH)
        ctx.events.emit(AppEvent.SCENE_CHANGED)