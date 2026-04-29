import os
from typing import List
from PySide6.QtWidgets import (QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, 
                               QFileDialog, QMessageBox, QMenu, QListWidgetItem, QListWidget)
from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor, QIcon

from src.ui.views.panels.base_panel import BasePanel
from src.ui.widgets.custom_lists import AssetListWidget

# Import SSOT configuration
from src.app.config import TEXTURE_CHANNELS, PANEL_TITLE_ASSET, DEFAULT_UI_MARGIN

class AssetBrowserPanelView(BasePanel):
    """
    Panel view for browsing, importing, and instantiating 3D models and textures.
    """
    PANEL_TITLE = PANEL_TITLE_ASSET
    PANEL_DOCK_AREA = Qt.BottomDockWidgetArea

    def setup_ui(self) -> None:
        """Constructs the horizontal split layout for Models and Textures."""
        self.layout = QHBoxLayout(self)
        
        m = DEFAULT_UI_MARGIN
        self.layout.setContentsMargins(m, m, m, m)

        # --- 3D Models Group ---
        group_m = QGroupBox("3D Models (Drag into Scene)")
        lay_m = QVBoxLayout(group_m)
        self.list_models = AssetListWidget("MODEL", self._controller)
        btn_m = QPushButton("Import Model (.obj, .ply)")
        btn_m.clicked.connect(self._on_import_model)
        lay_m.addWidget(btn_m)
        lay_m.addWidget(self.list_models)

        # --- Textures Group ---
        group_t = QGroupBox("Textures (Drag onto Entity)")
        lay_t = QVBoxLayout(group_t)
        self.list_tex = AssetListWidget("TEXTURE", self._controller)
        btn_t = QPushButton("Import Texture (.png, .jpg)")
        btn_t.clicked.connect(self._on_import_tex)
        lay_t.addWidget(btn_t)
        lay_t.addWidget(self.list_tex)

        self.layout.addWidget(group_m)
        self.layout.addWidget(group_t)

    def bind_events(self) -> None:
        """Wires up the double-click actions for immediate instantiation or assignment."""
        self.list_models.itemDoubleClicked.connect(self._on_model_double_clicked)
        self.list_tex.itemDoubleClicked.connect(self._on_texture_double_clicked)

    def _populate_list_widget(self, widget: QListWidget, file_paths: List[str], is_texture: bool = False) -> None:
        """Utility function to populate lists and handle duplicate filenames gracefully."""
        widget.clear()
        counts = {}
        for path in file_paths:
            base_name = os.path.basename(path)
            if base_name in counts:
                counts[base_name] += 1
                name, ext = os.path.splitext(base_name)
                disp_name = f"{name} ({counts[base_name]}){ext}"
            else:
                counts[base_name] = 0
                disp_name = base_name
                
            item = QListWidgetItem(disp_name)
            item.setToolTip(path)
            item.setData(Qt.UserRole, path)
            if is_texture and os.path.exists(path): 
                item.setIcon(QIcon(path))
            widget.addItem(item)

    def build_asset_lists(self, models: List[str], textures: List[str]) -> None:
        """Rebuilds both the model and texture lists based on the active project state."""
        self._populate_list_widget(self.list_models, models, is_texture=False)
        self._populate_list_widget(self.list_tex, textures, is_texture=True)

    def highlight_texture(self, path_to_find: str) -> None:
        """Highlights a specific texture in the list, typically triggered when an entity is selected."""
        self.list_tex.clearSelection()
        if not path_to_find: 
            return
        
        for i in range(self.list_tex.count()):
            item = self.list_tex.item(i)
            if item.data(Qt.UserRole) == path_to_find:
                item.setSelected(True)
                self.list_tex.scrollToItem(item)
                break

    def _on_import_model(self) -> None:
        """Spawns the system dialog to locate and cache a 3D geometry file."""
        path, _ = QFileDialog.getOpenFileName(self, "Import Model", "", "Models (*.obj *.ply)")
        if path and self._controller:
            self._controller.import_model(path)

    def _on_import_tex(self) -> None:
        """Spawns the system dialog to import an image texture."""
        path, _ = QFileDialog.getOpenFileName(self, "Import Texture", "", "Images (*.png *.jpg *.jpeg)")
        if path and self._controller:
            self._controller.import_texture(path)

    def _on_model_double_clicked(self, item: QListWidgetItem) -> None:
        """Directly dispatches the spawn command to the Controller."""
        if not self._controller: 
            return
        path = item.data(Qt.UserRole)
        self._controller.spawn_model(path)

    def _on_texture_double_clicked(self, item: QListWidgetItem) -> None:
        """Opens a contextual menu allowing the user to select which material channel to map the texture to."""
        if not self._controller: 
            return
        path = item.data(Qt.UserRole)
        
        menu = QMenu(self)
        menu.setStyleSheet("QMenu { font-size: 13px; padding: 5px; } QMenu::item { padding: 5px 20px; }")
        
        # Dynamically build Actions from the SSOT configuration dictionary
        for label, attr_name in TEXTURE_CHANNELS.items():
            action = menu.addAction(f"Set as {label}")
            action.setData(attr_name)
        
        action_selected = menu.exec(QCursor.pos())
        if not action_selected: 
            return
        
        map_attr = action_selected.data()
        self._controller.apply_texture(path, map_attr)