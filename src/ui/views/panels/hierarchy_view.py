from PySide6.QtWidgets import QVBoxLayout, QTreeWidgetItem, QTreeWidgetItemIterator, QStyle
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QKeyEvent
from typing import Any, Dict, Optional, List

# Inherit from standard MVC BasePanel
from src.ui.views.panels.base_panel import BasePanel

# Import the tree widget with Drag & Drop support from the widgets directory
from src.ui.widgets.custom_lists import EntityTreeWidget

# Import SSOT configuration
from src.app.config import PANEL_TITLE_HIERARCHY

class HierarchyPanelView(BasePanel):
    """
    Dumb View for the Scene Graph Hierarchy tree.
    Exclusively handles UI rendering and reports user actions (Clicks, Hotkeys, Drag & Drop) 
    to the HierarchyController.
    """
    # --- Metadata for MainController to automatically create Docks ---
    PANEL_TITLE = PANEL_TITLE_HIERARCHY
    PANEL_DOCK_AREA = Qt.LeftDockWidgetArea

    def setup_ui(self) -> None:
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Pass the controller directly to the TreeWidget to enable Drag & Drop reporting
        self.tree_widget = EntityTreeWidget(self._controller)
        self.layout.addWidget(self.tree_widget)

    def bind_events(self) -> None:
        # Report when the user clicks to select a different item
        self.tree_widget.currentItemChanged.connect(self._on_item_changed)
        
        # Set up the right-click Context Menu
        self.tree_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_widget.customContextMenuRequested.connect(self._on_context_menu)

    # =========================================================================
    # EVENT CAPTURE & DELEGATION (View -> Controller)
    # =========================================================================

    def keyPressEvent(self, e: QKeyEvent) -> None:
        """Catches keyboard shortcuts and translates them into corresponding Controller actions."""
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

    def _on_item_changed(self, current: Optional[QTreeWidgetItem], previous: Optional[QTreeWidgetItem]) -> None:
        """Extracts the ID of the selected Entity and reports it to the Controller."""
        if not self._controller: 
            return
        
        if current:
            idx = current.data(0, Qt.UserRole)
            if idx is not None:
                self._controller.handle_item_selected(idx)
        else:
            self._controller.handle_item_selected(-1)

    def _on_context_menu(self, pos: QPoint) -> None:
        """Converts relative coordinates to global coordinates for the Controller to draw the Popup Menu."""
        if self._controller:
            global_pos = self.tree_widget.mapToGlobal(pos)
            self._controller.show_context_menu(global_pos)

    # =========================================================================
    # PUBLIC API FOR DATA INJECTION (Controller -> View)
    # =========================================================================

    def build_tree(self, entities_data: List[Dict[str, Any]], selected_idx: int) -> None:
        """
        Reconstructs the entire visual hierarchy tree based on the provided data list.
        Blocks signals to prevent infinite loops during UI updates.
        """
        self.tree_widget.blockSignals(True)
        self.tree_widget.clear()
        
        items_map: Dict[int, QTreeWidgetItem] = {}
        
        dir_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon)
        file_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon)
        
        # Phase 1: Instantiate all QTreeWidgetItems
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
            
        # Phase 2: Establish Parent-Child relationships based on the 'parent' attribute
        for data in entities_data:
            item = items_map[data["id"]]
            parent_id = data["parent"]
            
            if parent_id is not None and parent_id in items_map:
                items_map[parent_id].addChild(item)
            else:
                self.tree_widget.addTopLevelItem(item)
                
        self.tree_widget.expandAll()
        
        # Phase 3: Restore previous selection highlight state
        if selected_idx >= 0 and selected_idx in items_map:
            self.tree_widget.setCurrentItem(items_map[selected_idx])
            items_map[selected_idx].setSelected(True)
        else:
            self.tree_widget.setCurrentItem(None)
            
        self.tree_widget.blockSignals(False)

    def update_selection(self, idx: int) -> None:
        """
        Updates the UI highlight when an item is selected from an external source 
        (e.g., the user clicks directly on a model in the 3D Viewport).
        """
        self.tree_widget.blockSignals(True)
        
        # Clear current selection to prevent UI ghosting artifacts
        self.tree_widget.clearSelection()
        self.tree_widget.setCurrentItem(None)
        
        # Traverse the tree to locate and highlight the matching ID
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