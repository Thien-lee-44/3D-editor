from PySide6.QtWidgets import QVBoxLayout, QScrollArea, QWidget
from PySide6.QtCore import Qt
from typing import Any, Dict

from src.ui.views.panels.base_panel import BasePanel
from src.ui.widgets.inspector.header_widget import HeaderWidget
from src.ui.widgets.inspector.transform_widget import TransformWidget
from src.ui.widgets.inspector.mesh_widget import MeshWidget
from src.ui.widgets.inspector.light_widget import LightWidget
from src.ui.widgets.inspector.camera_widget import CameraWidget

# Import SSOT configuration
from src.app.config import (
    PANEL_TITLE_INSPECTOR, 
    PANEL_MIN_WIDTH_INSPECTOR, 
    PANEL_CONTENT_MIN_WIDTH, 
    DEFAULT_UI_MARGIN
)

class InspectorPanelView(BasePanel):
    """
    Dumb View that aggregates specific component property widgets based on the selected entity.
    """
    PANEL_TITLE = PANEL_TITLE_INSPECTOR
    PANEL_DOCK_AREA = Qt.RightDockWidgetArea
    PANEL_MIN_WIDTH = PANEL_MIN_WIDTH_INSPECTOR

    def setup_ui(self) -> None:
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff) 
        
        self.content = QWidget()
        self.content.setMinimumWidth(PANEL_CONTENT_MIN_WIDTH)
        content_layout = QVBoxLayout(self.content)
        
        m = DEFAULT_UI_MARGIN
        content_layout.setContentsMargins(m, m, m, m)
        content_layout.setAlignment(Qt.AlignTop)

        # Pass the controller to the sub-widgets so they can emit changes
        self.header_widget = HeaderWidget(self._controller)
        self.transform_widget = TransformWidget(self._controller)
        self.mesh_widget = MeshWidget(self._controller)
        self.light_widget = LightWidget(self._controller)
        self.camera_widget = CameraWidget(self._controller)

        content_layout.addWidget(self.header_widget)
        content_layout.addWidget(self.transform_widget)
        content_layout.addWidget(self.mesh_widget)
        content_layout.addWidget(self.light_widget)
        content_layout.addWidget(self.camera_widget)
        content_layout.addStretch(1)
        
        scroll.setWidget(self.content)
        self.layout.addWidget(scroll)
        
        self.hide_all_components()

    def hide_all_components(self) -> None:
        for g in [self.header_widget, self.transform_widget, self.mesh_widget, self.light_widget, self.camera_widget]: 
            g.setVisible(False)

    # =========================================================================
    # PUBLIC API FOR DATA INJECTION
    # =========================================================================

    def update_inspector_data(self, data: Dict[str, Any]) -> None:
        """Parses the data dictionary and delegates rendering to sub-widgets."""
        has_tf = bool(data.get("tf"))
        has_light = bool(data.get("light"))
        has_cam = bool(data.get("cam"))
        has_mesh = bool(data.get("mesh") and not has_light and not has_cam)
        mesh_visible = data["mesh"]["visible"] if data.get("mesh") else True

        self.header_widget.update_data(data["name"])
        
        if has_tf: 
            self.transform_widget.update_data(data["tf"], has_light, data["light"]["type"] if has_light else "")
        if has_mesh: 
            self.mesh_widget.update_data(data["mesh"])
        if has_light: 
            self.light_widget.update_data(data["light"], mesh_visible)
        if has_cam: 
            self.camera_widget.update_data(data["cam"], mesh_visible)

        self.header_widget.setVisible(True)
        self.transform_widget.setVisible(has_tf)
        self.mesh_widget.setVisible(has_mesh)
        self.light_widget.setVisible(has_light)
        self.camera_widget.setVisible(has_cam)

    def fast_update_transform(self, transform_tuple: tuple) -> None:
        """Anti-lag hook: Extracts (mode, values) from tuple and updates the correct widget."""
        if not isinstance(transform_tuple, tuple) or len(transform_tuple) != 2:
            return
            
        mode, values = transform_tuple
        
        # Calls down to the Transform Widget to update exactly one coordinate axis
        self.transform_widget.fast_update_single_axis(mode, values)
        
        # If currently rotating (ROTATE) and the Light panel is open, sync the Pitch/Yaw sliders
        if mode == "ROTATE" and self.light_widget.isVisible():
            if hasattr(self.light_widget, 'fast_update_rotation'):
                self.light_widget.fast_update_rotation(values)