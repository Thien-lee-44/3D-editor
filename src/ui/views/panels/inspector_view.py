import glm
from PySide6.QtWidgets import QVBoxLayout, QScrollArea, QWidget
from PySide6.QtCore import Qt
from typing import Any, Dict

from src.app import ctx
from src.ui.views.panels.base_panel import BasePanel
from src.ui.widgets.inspector.header_widget import HeaderWidget
from src.ui.widgets.inspector.transform_widget import TransformWidget
from src.ui.widgets.inspector.mesh_widget import MeshWidget
from src.ui.widgets.inspector.light_widget import LightWidget
from src.ui.widgets.inspector.camera_widget import CameraWidget
from src.ui.widgets.inspector.semantic_widget import SemanticWidget
from src.ui.widgets.inspector.animation_widget import AnimationWidget

from src.app.config import (
    PANEL_TITLE_INSPECTOR, 
    PANEL_MIN_WIDTH_INSPECTOR, 
    PANEL_CONTENT_MIN_WIDTH, 
    DEFAULT_UI_MARGIN
)

class InspectorPanelView(BasePanel):
    """Orchestrates data routing between Live World State and the Dynamic Keyframe Bag."""
    PANEL_TITLE = PANEL_TITLE_INSPECTOR
    PANEL_DOCK_AREA = Qt.RightDockWidgetArea
    PANEL_MIN_WIDTH = PANEL_MIN_WIDTH_INSPECTOR

    def setup_ui(self) -> None:
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QScrollArea.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff) 
        
        self.content = QWidget()
        self.content.setMinimumWidth(PANEL_CONTENT_MIN_WIDTH)
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(DEFAULT_UI_MARGIN, DEFAULT_UI_MARGIN, DEFAULT_UI_MARGIN, DEFAULT_UI_MARGIN)
        self.content_layout.setAlignment(Qt.AlignTop)

        self.header_widget = HeaderWidget(self._controller)
        self.semantic_widget = SemanticWidget(self._controller)
        self.animation_widget = AnimationWidget(self._controller)
        self.transform_widget = TransformWidget(self._controller)
        self.mesh_widget = MeshWidget(self._controller)
        self.light_widget = LightWidget(self._controller)
        self.camera_widget = CameraWidget(self._controller)

        self.content_layout.addWidget(self.header_widget)
        self.content_layout.addWidget(self.semantic_widget)
        self.content_layout.addWidget(self.animation_widget)
        self.content_layout.addWidget(self.transform_widget)
        self.content_layout.addWidget(self.mesh_widget)
        self.content_layout.addWidget(self.light_widget)
        self.content_layout.addWidget(self.camera_widget)
        self.content_layout.addStretch(1)
        
        self.scroll.setWidget(self.content)
        self.layout.addWidget(self.scroll)
        self.hide_all_components()

    def hide_all_components(self) -> None:
        self.content.setStyleSheet("")
        for w in [self.header_widget, self.semantic_widget, self.animation_widget, 
                  self.transform_widget, self.mesh_widget, self.light_widget, self.camera_widget]:
            w.setVisible(False)

    def update_inspector_data(self, data: Dict[str, Any]) -> None:
        kf_idx = data.get("active_keyframe_index", -1)
        anim_data = data.get("anim", {})
        keyframes = anim_data.get("keyframes", [])
        
        has_tf = bool(data.get("tf"))
        has_light = bool(data.get("light"))
        has_cam = bool(data.get("cam"))
        has_semantic = bool(data.get("semantic"))
        has_anim = bool(data.get("anim"))
        
        # Hide MeshWidget if the entity is a Light or Camera (proxy meshes should not be edited)
        has_mesh = bool(data.get("mesh")) and not has_light and not has_cam
        
        tf_data = data.get("tf", {})
        light_data = data.get("light", {})
        mesh_data = data.get("mesh", {})
        cam_data = data.get("cam", {})
        
        # KEYFRAME BAG EXTRACTION
        if 0 <= kf_idx < len(keyframes):
            kf_state = keyframes[kf_idx].get("state", {})
            
            if "Transform" in kf_state:
                snap_tf = kf_state["Transform"]
                if "position" in snap_tf: tf_data["position"] = snap_tf["position"]
                if "scale" in snap_tf: tf_data["scale"] = snap_tf["scale"]
                if "rotation" in snap_tf: tf_data["rotation"] = snap_tf["rotation"]
                if "quat_rot" in snap_tf: tf_data["quat_rot"] = snap_tf["quat_rot"]
                    
            if "Light" in kf_state:
                light_data = dict(data.get("light", {}))
                for key, val in kf_state["Light"].items():
                    light_data[key] = val
                        
            if "Mesh" in kf_state:
                mesh_data = dict(data.get("mesh", {}))
                for key, val in kf_state["Mesh"].items():
                    mesh_data[key] = val

            if "Camera" in kf_state:
                cam_data = dict(data.get("cam", {}))
                for key, val in kf_state["Camera"].items():
                    if key == "active":
                        cam_data["is_active"] = val
                    elif key == "ortho":
                        cam_data["ortho_size"] = val
                    else:
                        cam_data[key] = val

        self.header_widget.update_data(data.get("name", "Entity"))
        self.header_widget.setVisible(True)
        
        if has_semantic:
            self.semantic_widget.update_data(data["semantic"])
            self.semantic_widget.setVisible(True)
        else:
            self.semantic_widget.setVisible(False)
            
        if has_anim:
            anim_payload = dict(anim_data)
            anim_payload["active_keyframe_index"] = kf_idx
            anim_payload["active_keyframe_time"] = data.get("active_keyframe_time", 0.0)
            self.animation_widget.update_data(anim_payload)
            self.animation_widget.setVisible(True)
        else:
            self.animation_widget.setVisible(False)
            
        if has_tf: 
            # Simplified update call: transform_widget now parses locked_axes directly from tf_data
            self.transform_widget.update_data(tf_data)
            self.transform_widget.setVisible(True)
        else:
            self.transform_widget.setVisible(False)
            
        if has_mesh: 
            self.mesh_widget.update_data(mesh_data)
            self.mesh_widget.setVisible(True)
        else:
            self.mesh_widget.setVisible(False)
            
        if has_light: 
            m_vis = data["mesh"]["visible"] if data.get("mesh") else True
            self.light_widget.update_data(light_data, m_vis)
            self.light_widget.setVisible(True)
        else:
            self.light_widget.setVisible(False)
            
        if has_cam: 
            m_vis = data["mesh"]["visible"] if data.get("mesh") else True
            self.camera_widget.update_data(cam_data, m_vis)
            self.camera_widget.setVisible(True)
        else:
            self.camera_widget.setVisible(False)

    def fast_update_transform(self, transform_tuple: tuple) -> None:
        if not isinstance(transform_tuple, tuple) or len(transform_tuple) != 2: return
        mode, values = transform_tuple
        self.transform_widget.fast_update_single_axis(mode, values)
        if mode == "ROTATE" and self.light_widget.isVisible():
            if hasattr(self.light_widget, 'fast_update_rotation'):
                data = ctx.engine.get_selected_entity_data()
                if data and data.get("light"):
                    pitch = data["light"].get("pitch", 0.0)
                    yaw = data["light"].get("yaw", 0.0)
                    self.light_widget.fast_update_rotation((pitch, yaw))
