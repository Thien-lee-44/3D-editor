from PySide6.QtWidgets import QVBoxLayout, QTreeWidgetItem, QTreeWidgetItemIterator, QStyle
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QKeyEvent
from typing import Any, Dict, Optional, List, Set

from src.ui.views.panels.base_panel import BasePanel
from src.ui.widgets.custom_lists import EntityTreeWidget
from src.app.config import PANEL_TITLE_HIERARCHY

class HierarchyPanelView(BasePanel):
    """
    Dumb View for the Scene Graph Hierarchy tree.
    Exclusively handles UI rendering, state preservation, and reports user actions.
    """
    PANEL_TITLE = PANEL_TITLE_HIERARCHY
    PANEL_DOCK_AREA = Qt.LeftDockWidgetArea

    def setup_ui(self) -> None:
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.tree_widget = EntityTreeWidget(self._controller)
        self.layout.addWidget(self.tree_widget)
        
        self._is_internal_selection: bool = False
        self._is_updating_externally: bool = False

    def bind_events(self) -> None:
        self.tree_widget.itemSelectionChanged.connect(self._on_selection_changed)
        self.tree_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_widget.customContextMenuRequested.connect(self._on_context_menu)

    # =========================================================================
    # EVENT CAPTURE & DELEGATION
    # =========================================================================

    def keyPressEvent(self, e: QKeyEvent) -> None:
        if not self._controller:
            super().keyPressEvent(e)
            return
            
        ctrl = bool(e.modifiers() & Qt.ControlModifier)
        
        if ctrl and e.key() == Qt.Key_C: 
            self._controller.handle_copy()
        elif ctrl and e.key() == Qt.Key_X: 
            self._controller.handle_cut()
        elif ctrl and e.key() == Qt.Key_V: 
            self._controller.handle_paste()
        elif e.key() in [Qt.Key_Delete, Qt.Key_Backspace]: 
            self._controller.handle_delete()
        else: 
            super().keyPressEvent(e)

    def _on_selection_changed(self) -> None:
        if not self._controller: 
            return
        if self._is_updating_externally:
            return
            
        items = self.tree_widget.selectedItems()
        
        self._is_internal_selection = True
        
        if not items:
            self._controller.handle_item_selected(-1)
            if hasattr(self._controller, 'handle_multi_selection'):
                self._controller.handle_multi_selection([])
        else:
            primary_item = self.tree_widget.currentItem()
            if primary_item and primary_item.isSelected():
                primary_idx = primary_item.data(0, Qt.UserRole)
            else:
                primary_idx = items[0].data(0, Qt.UserRole)
                
            self._controller.handle_item_selected(primary_idx)
            
            if hasattr(self._controller, 'handle_multi_selection'):
                all_ids = [i.data(0, Qt.UserRole) for i in items]
                self._controller.handle_multi_selection(all_ids)
                
        self._is_internal_selection = False

    def _on_context_menu(self, pos: QPoint) -> None:
        if self._controller and hasattr(self._controller, 'show_context_menu'):
            global_pos = self.tree_widget.mapToGlobal(pos)
            self._controller.show_context_menu(global_pos)

    # =========================================================================
    # PUBLIC API FOR DATA INJECTION
    # =========================================================================

    def build_tree(self, entities_data: List[Dict[str, Any]], selected_idx: int) -> None:
        self._is_updating_externally = True
        self.tree_widget.blockSignals(True)
        
        expanded_ids: Set[int] = set()
        is_first_load = self.tree_widget.topLevelItemCount() == 0
        
        iterator = QTreeWidgetItemIterator(self.tree_widget)
        while iterator.value():
            item = iterator.value()
            if item.isExpanded():
                ent_id = item.data(0, Qt.UserRole)
                if ent_id is not None:
                    expanded_ids.add(ent_id)
            iterator += 1

        self.tree_widget.clear()
        
        items_map: Dict[int, QTreeWidgetItem] = {}
        
        dir_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon)
        file_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon)
        
        for data in entities_data:
            item = QTreeWidgetItem([data["name"]])
            item.setData(0, Qt.UserRole, data["id"])
            item.setFlags(item.flags() | Qt.ItemIsEditable | Qt.ItemIsDragEnabled | Qt.ItemIsDropEnabled)
            
            if data["is_group"]:
                item.setFlags(item.flags() | Qt.ItemIsDropEnabled)
                item.setIcon(0, dir_icon)
            else:
                item.setFlags(item.flags() & ~Qt.ItemIsDropEnabled)
                item.setIcon(0, file_icon)
                
            items_map[data["id"]] = item
            
        for data in entities_data:
            item = items_map[data["id"]]
            parent_id = data["parent"]
            
            if parent_id is not None and parent_id in items_map:
                items_map[parent_id].addChild(item)
            else:
                self.tree_widget.addTopLevelItem(item)
                
        if is_first_load:
            self.tree_widget.expandAll()
        else:
            for data in entities_data:
                if data["id"] in expanded_ids:
                    items_map[data["id"]].setExpanded(True)
        
        if selected_idx >= 0 and selected_idx in items_map:
            self.tree_widget.setCurrentItem(items_map[selected_idx])
            items_map[selected_idx].setSelected(True)
        else:
            self.tree_widget.setCurrentItem(None)
            
        self.tree_widget.blockSignals(False)
        self._is_updating_externally = False

    def update_selection(self, idx: int) -> None:
        if self._is_internal_selection:
            return
            
        self._is_updating_externally = True
        self.tree_widget.blockSignals(True)
        
        self.tree_widget.clearSelection()
        self.tree_widget.setCurrentItem(None)
        
        if idx >= 0:
            iterator = QTreeWidgetItemIterator(self.tree_widget)
            while iterator.value():
                item = iterator.value()
                if item.data(0, Qt.UserRole) == idx:
                    self.tree_widget.setCurrentItem(item)
                    item.setSelected(True)
                    break
                iterator += 1
                
        self.tree_widget.blockSignals(False)
        self._is_updating_externally = False