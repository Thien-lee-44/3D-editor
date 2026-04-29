from typing import Any, Dict, Optional, List, Set

from PySide6.QtWidgets import QVBoxLayout, QTreeWidgetItem, QStyle, QApplication
from PySide6.QtCore import Qt, QPoint, QEvent
from PySide6.QtGui import QKeyEvent, QWheelEvent, QCursor

from src.ui.views.panels.base_panel import BasePanel
from src.ui.widgets.custom_lists import EntityTreeWidget
from src.app.config import PANEL_TITLE_HIERARCHY

class HierarchyPanelView(BasePanel):
    """
    Dumb View for the Scene Graph Hierarchy tree.
    Exclusively handles UI rendering, state preservation, and reports user actions.
    Utilizes a global event filter to bypass Qt's QDrag loop, ensuring smooth
    mouse-wheel scrolling and edge auto-scrolling during drag-and-drop operations.
    """
    PANEL_TITLE = PANEL_TITLE_HIERARCHY
    PANEL_DOCK_AREA = Qt.LeftDockWidgetArea

    def setup_ui(self) -> None:
        """Initializes the layout and custom Tree Widget."""
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.tree_widget = EntityTreeWidget(self._controller)
        
        # Enable smooth automatic scrolling when dragging near the top/bottom edges
        self.tree_widget.setAutoScroll(True)
        self.tree_widget.setAutoScrollMargin(32)
        
        # Install an Application-level event filter to intercept wheel events 
        # before the QDrag modal loop consumes them.
        QApplication.instance().installEventFilter(self)
        
        self.layout.addWidget(self.tree_widget)
        
        self._is_internal_selection: bool = False
        self._is_updating_externally: bool = False

    def bind_events(self) -> None:
        """Wires up local UI signals to event handlers."""
        self.tree_widget.itemSelectionChanged.connect(self._on_selection_changed)
        self.tree_widget.itemChanged.connect(self._on_item_changed)
        self.tree_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_widget.customContextMenuRequested.connect(self._on_context_menu)

    # =========================================================================
    # EVENT CAPTURE & DELEGATION
    # =========================================================================

    def eventFilter(self, obj: Any, event: QEvent) -> bool:
        """
        Global Event Interceptor.
        Forces the Hierarchy tree to scroll if the mouse wheel is used while hovering 
        over it, even if a Drag-and-Drop operation is currently locking the event queue.
        """
        if event.type() == QEvent.Type.Wheel:
            # Map global cursor position to local tree widget coordinates
            local_pos = self.tree_widget.mapFromGlobal(QCursor.pos())
            
            # Verify if the mouse is currently hovering over the tree widget
            if self.tree_widget.rect().contains(local_pos):
                try:
                    delta = event.angleDelta().y()
                    if delta != 0:
                        scroll_bar = self.tree_widget.verticalScrollBar()
                        # Use a smooth scrolling step (e.g., 3 steps per notch)
                        step = scroll_bar.singleStep() * 3
                        direction = -1 if delta > 0 else 1
                        scroll_bar.setValue(scroll_bar.value() + direction * step)
                    return True  # Consume the event to prevent double-scrolling
                except AttributeError:
                    pass
                    
        return super().eventFilter(obj, event)

    def keyPressEvent(self, e: QKeyEvent) -> None:
        """Binds standard keyboard shortcuts (Copy, Cut, Paste, Delete)."""
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
        """Notifies the Controller when the user clicks an entity node."""
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

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        """Fires when the user completes inline editing of a tree item."""
        if self._is_updating_externally or not self._controller:
            return
            
        ent_id = item.data(0, Qt.UserRole)
        new_name = item.text(0).strip()
        
        if new_name and ent_id is not None:
            self._controller.handle_rename(ent_id, new_name)

    def _on_context_menu(self, pos: QPoint) -> None:
        """Spawns the right-click contextual menu via the Controller."""
        if self._controller and hasattr(self._controller, 'show_context_menu'):
            global_pos = self.tree_widget.mapToGlobal(pos)
            self._controller.show_context_menu(global_pos)

    # =========================================================================
    # PUBLIC API FOR DATA INJECTION
    # =========================================================================

    def build_tree(self, entities_data: List[Dict[str, Any]], selected_idx: int) -> None:
        """
        Reconstructs the hierarchical tree representation based on backend state.
        Maintains expanded states and selection dynamically.
        """
        self._is_updating_externally = True
        self.tree_widget.blockSignals(True)
        
        self.tree_widget.clearSelection()
        self.tree_widget.setCurrentItem(None)
        
        expanded_ids: Set[int] = set()
        is_first_load = self.tree_widget.topLevelItemCount() == 0
        
        def cache_expanded(item: QTreeWidgetItem) -> None:
            if item.isExpanded():
                ent_id = item.data(0, Qt.UserRole)
                if ent_id is not None:
                    expanded_ids.add(ent_id)
            for i in range(item.childCount()):
                cache_expanded(item.child(i))

        for i in range(self.tree_widget.topLevelItemCount()):
            cache_expanded(self.tree_widget.topLevelItem(i))

        self.tree_widget.clear()
        items_map: Dict[int, QTreeWidgetItem] = {}
        
        dir_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon)
        file_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon)
        
        # First pass: Create all items
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
            
        # Second pass: Parent-Child relationships
        for data in entities_data:
            item = items_map[data["id"]]
            parent_id = data["parent"]
            
            if parent_id is not None and parent_id in items_map:
                items_map[parent_id].addChild(item)
            else:
                self.tree_widget.addTopLevelItem(item)
                
        # Restore expansion state
        if is_first_load:
            self.tree_widget.expandAll()
        else:
            for data in entities_data:
                if data["id"] in expanded_ids:
                    items_map[data["id"]].setExpanded(True)
        
        # Restore selection state
        if selected_idx >= 0 and selected_idx in items_map:
            self.tree_widget.setCurrentItem(items_map[selected_idx])
            items_map[selected_idx].setSelected(True)
        else:
            self.tree_widget.setCurrentItem(None)
            
        self.tree_widget.blockSignals(False)
        self._is_updating_externally = False

    def update_selection(self, idx: int) -> None:
        """Silently synchronizes the UI selection without triggering events."""
        if self._is_internal_selection:
            return
            
        self._is_updating_externally = True
        self.tree_widget.blockSignals(True)
        
        self.tree_widget.clearSelection()
        self.tree_widget.setCurrentItem(None)
        
        if idx >= 0:
            def find_and_select(item: QTreeWidgetItem) -> bool:
                if item.data(0, Qt.UserRole) == idx:
                    self.tree_widget.setCurrentItem(item)
                    item.setSelected(True)
                    return True
                for i in range(item.childCount()):
                    if find_and_select(item.child(i)):
                        return True
                return False

            for i in range(self.tree_widget.topLevelItemCount()):
                if find_and_select(self.tree_widget.topLevelItem(i)):
                    break
                
        self.tree_widget.blockSignals(False)
        self._is_updating_externally = False