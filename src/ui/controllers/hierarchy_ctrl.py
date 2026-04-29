"""
Hierarchy Controller.

Manages interactions with the Scene Graph Tree View.
Handles entity selection, renaming, drag-and-drop reparenting, and context menu actions.
"""

from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QMessageBox
from typing import List, Dict, Optional

from src.app import ctx, AppEvent
from src.ui.error_handler import safe_execute
from src.ui.views.panels.hierarchy_view import HierarchyPanelView


class HierarchyController:
    """Coordinates data flow and user actions for the Hierarchy Panel."""

    def __init__(self) -> None:
        self.view = HierarchyPanelView(controller=self)
        
        self.selected_multi_ids: List[int] = [] 
        
        ctx.events.subscribe(AppEvent.HIERARCHY_NEEDS_REFRESH, self.refresh_view)
        ctx.events.subscribe(AppEvent.ENTITY_SELECTED, self.on_global_selection)

    @safe_execute(context="Refresh Hierarchy")
    def refresh_view(self) -> None:
        """Fetches the latest Scene Graph topology and rebuilds the UI Tree."""
        entities_data = ctx.engine.get_scene_entities_list()
        selected_idx = ctx.engine.get_selected_entity_id()
        self.view.build_tree(entities_data, selected_idx)

    def on_global_selection(self, entity_id: int) -> None:
        """Responds to global selection changes (e.g., from Viewport picking)."""
        self.view.update_selection(entity_id)

    @safe_execute(context="Select Entity")
    def handle_item_selected(self, entity_id: int) -> None:
        """Updates the Engine state when a user clicks an item in the Tree View."""
        ctx.engine.select_entity(entity_id)
        ctx.events.emit(AppEvent.ENTITY_SELECTED, entity_id)
        ctx.events.emit(AppEvent.SCENE_CHANGED)

    def handle_multi_selection(self, ids: List[int]) -> None:
        """Stores the list of currently selected entity IDs for batch operations."""
        self.selected_multi_ids = ids

    @safe_execute(context="Reorder Hierarchy")
    def handle_hierarchy_reorder(self, hierarchy_mapping: Dict[int, Optional[int]]) -> None:
        """Processes drag-and-drop reparenting operations."""
        ctx.engine.sync_hierarchy_from_ui(hierarchy_mapping)
        ctx.events.emit(AppEvent.HIERARCHY_NEEDS_REFRESH)
        ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Rename Entity")
    def handle_rename(self, entity_id: int, new_name: str) -> None:
        """Validates and applies a new name to an entity, ensuring uniqueness."""
        new_name = new_name.strip()
        if not new_name:
            ctx.events.emit(AppEvent.HIERARCHY_NEEDS_REFRESH)
            return

        # 1. Unique name validation (logic from HeaderWidget)
        entities = ctx.engine.get_scene_entities_list()
        for ent in entities:
            if ent["id"] != entity_id and ent["name"] == new_name:
                QMessageBox.warning(
                    self.view, 
                    "Invalid Name", 
                    f"The name '{new_name}' is already in use.\nPlease choose a unique name."
                )
                ctx.events.emit(AppEvent.HIERARCHY_NEEDS_REFRESH)
                return

        ctx.events.emit(AppEvent.ACTION_BEFORE_MUTATION)
        
        # 2. Synchronized property update logic
        # Temporarily select to use the selection-based set_component_properties
        prev_selected_id = ctx.engine.get_selected_entity_id()
        ctx.engine.select_entity(entity_id)
        ctx.engine.set_component_properties("Entity", {"name": new_name})
        ctx.engine.select_entity(prev_selected_id)
            
        if prev_selected_id == entity_id:
            ctx.events.emit(AppEvent.COMPONENT_PROPERTY_CHANGED)
            
        ctx.events.emit(AppEvent.SCENE_CHANGED)
        ctx.events.emit(AppEvent.HIERARCHY_NEEDS_REFRESH)

    def show_context_menu(self, global_pos: QPoint) -> None:
        """Delegates Right-Click context menu rendering to the Main Window."""
        if hasattr(ctx, 'main_window') and hasattr(ctx.main_window, 'show_context_menu'):
            ctx.main_window.show_context_menu(global_pos)

    @safe_execute(context="Copy Entity")
    def handle_copy(self) -> None:
        """Copies the currently selected entity into the Engine's memory buffer."""
        ctx.engine.copy_selected()

    @safe_execute(context="Cut Entity")
    def handle_cut(self) -> None:
        """Copies the selected entity and immediately removes it from the Scene."""
        ctx.events.emit(AppEvent.ACTION_BEFORE_MUTATION) 
        ctx.engine.cut_selected()
        ctx.events.emit(AppEvent.HIERARCHY_NEEDS_REFRESH)
        ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Paste Entity")
    def handle_paste(self) -> None:
        """Instantiates the clipboard buffer back into the Scene Graph."""
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
        """Safely removes the selected entity from the Scene."""
        ctx.events.emit(AppEvent.ACTION_BEFORE_MUTATION)
        ctx.engine.delete_selected()
        ctx.events.emit(AppEvent.HIERARCHY_NEEDS_REFRESH)
        ctx.events.emit(AppEvent.SCENE_CHANGED)