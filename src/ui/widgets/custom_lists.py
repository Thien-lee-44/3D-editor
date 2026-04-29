"""
Custom List & Tree Widgets.
Implements specialized item views for the Asset Browser and Hierarchy Panel,
supporting internal Drag-and-Drop operations and context menus.
"""

import os
from typing import Any, List, Dict, Optional
from PySide6.QtWidgets import (QListWidget, QListWidgetItem, QMenu, QMessageBox,
                               QTreeWidget, QTreeWidgetItem, QAbstractItemView)
from PySide6.QtCore import Qt, QMimeData, QSize, QPoint
from PySide6.QtGui import QDropEvent, QMouseEvent

from src.app.config import ASSET_ICON_SIZE, ASSET_LIST_SPACING, CONTEXT_MENU_STYLE

# =========================================================================
# ASSET BROWSER WIDGETS
# =========================================================================

class AssetListWidget(QListWidget):
    """
    Specialized List Widget for displaying resources (Models/Textures).
    Packages data into MIME strings so the 3D Viewport can intercept Drag-and-Drop events.
    """
    
    def __init__(self, asset_type: str, controller: Any) -> None:
        super().__init__()
        self.asset_type: str = asset_type 
        self._controller: Any = controller 
        
        self.setDragEnabled(True)
        self.setDefaultDropAction(Qt.CopyAction)
        self.setSelectionMode(QListWidget.SingleSelection)
        
        # Enlarge display format for Texture images
        if asset_type == 'TEXTURE':
            self.setIconSize(QSize(ASSET_ICON_SIZE, ASSET_ICON_SIZE))
            self.setSpacing(ASSET_LIST_SPACING)
            
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

    def mimeData(self, items: List[QListWidgetItem]) -> QMimeData:
        """
        Encapsulates the Asset path into QMimeData.
        Format: "TYPE|Absolute_Path" (e.g., "MODEL|C:/assets/car.obj").
        """
        mime = QMimeData()
        if items:
            path = items[0].data(Qt.UserRole)
            mime.setText(f"{self.asset_type}|{path}")
        return mime

    def _show_context_menu(self, pos: QPoint) -> None:
        """Displays a context menu for asset deletion."""
        item = self.itemAt(pos)
        if not item: 
            return
        
        path = item.data(Qt.UserRole)
        
        menu = QMenu(self)
        menu.setStyleSheet(CONTEXT_MENU_STYLE)
        
        action_delete = menu.addAction("Delete Asset")
        action_selected = menu.exec(self.mapToGlobal(pos))
        
        if action_selected == action_delete:
            ans = QMessageBox.question(
                self, "Confirm Delete", 
                f"Are you sure you want to remove this asset from the project?\n{os.path.basename(path)}",
                QMessageBox.Yes | QMessageBox.No
            )
            if ans == QMessageBox.Yes:
                self._controller.request_delete_asset(path, self.asset_type)


# =========================================================================
# HIERARCHY TREE WIDGET
# =========================================================================

class EntityTreeWidget(QTreeWidget):
    """
    Custom QTreeWidget supporting hierarchical drag-and-drop and multi-selection.
    """
    
    def __init__(self, controller: Any) -> None:
        super().__init__()
        self._controller: Any = controller 
        
        self.setHeaderHidden(True)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        
        # Enable Qt's internal Drag-and-Drop mechanics
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QAbstractItemView.InternalMove)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Safely handles multi-selection modifiers during Drag-and-Drop interactions."""
        modifiers = event.modifiers()
        if modifiers & (Qt.ControlModifier | Qt.ShiftModifier):
            self.setDragDropMode(QAbstractItemView.NoDragDrop)
            super().mousePressEvent(event)
            self.setDragDropMode(QAbstractItemView.InternalMove)
        else:
            super().mousePressEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        """
        Overrides the Drop event to synchronize the visual tree changes 
        with the underlying Engine's ECS.
        """
        # 1. Allow Qt to physically move the item within the UI tree
        super().dropEvent(event)
        
        hierarchy_mapping: Dict[int, Optional[int]] = {}
        
        # 2. Recursively traverse the UI to map relationships (Entity ID -> Parent ID)
        def traverse(item: QTreeWidgetItem, parent_id: Optional[int]) -> None:
            ent_id = item.data(0, Qt.UserRole)
            if ent_id is not None:
                hierarchy_mapping[ent_id] = parent_id
            for i in range(item.childCount()):
                traverse(item.child(i), ent_id)
                
        for i in range(self.topLevelItemCount()):
            traverse(self.topLevelItem(i), None)
            
        # 3. Dispatch the comprehensive mapping array to the Controller
        if self._controller and hasattr(self._controller, 'handle_hierarchy_reorder'):
            self._controller.handle_hierarchy_reorder(hierarchy_mapping)