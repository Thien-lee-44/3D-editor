from typing import Any, Dict
from PySide6.QtWidgets import QPushButton
from src.ui.widgets.custom_inputs import create_vec3_input
from .base_widget import BaseComponentWidget, set_vec3_spinboxes

# Import SSOT configuration
from src.app.config import (
    STYLE_BTN_RESET, DEFAULT_SPAWN_SCALE,
    TRANSFORM_POS_STEP, TRANSFORM_ROT_RANGE, TRANSFORM_ROT_STEP, 
    TRANSFORM_SCL_MIN, TRANSFORM_SCL_STEP
)

class TransformWidget(BaseComponentWidget):
    def __init__(self, controller: Any) -> None:
        # Call BaseComponentWidget and assign local _controller for safety
        super().__init__("Transform", controller)
        self._controller = controller 
        
        self.btn_reset = QPushButton("Reset Transform")
        self.btn_reset.setStyleSheet(STYLE_BTN_RESET)
        self.btn_reset.clicked.connect(self.reset_transform)
        self.layout.addWidget(self.btn_reset)
        
        # Call request_undo via Controller
        self.w_pos, self.sp_pos = create_vec3_input("Position:", self.apply_transform, step=TRANSFORM_POS_STEP, press_callback=self.request_undo)
        self.w_rot, self.sp_rot = create_vec3_input("Rotation:", self.apply_transform, min_val=TRANSFORM_ROT_RANGE[0], max_val=TRANSFORM_ROT_RANGE[1], step=TRANSFORM_ROT_STEP, press_callback=self.request_undo)
        self.w_scl, self.sp_scl = create_vec3_input("Scale:", self.apply_transform, default=DEFAULT_SPAWN_SCALE[0], min_val=TRANSFORM_SCL_MIN, step=TRANSFORM_SCL_STEP, press_callback=self.request_undo)
        
        self.layout.addWidget(self.w_pos)
        self.layout.addWidget(self.w_rot)
        self.layout.addWidget(self.w_scl)

    def request_undo(self) -> None:
        """Intercepts input interactions to dispatch state snapshots to the Undo stack."""
        if self._controller:
            self._controller.request_undo_snapshot()

    def update_data(self, tf_data: Dict[str, Any], has_light: bool, light_type: str) -> None:
        self.w_pos.setVisible(True)
        self.w_rot.setVisible(True)
        self.w_scl.setVisible(True)

        # Contextual UI logic: Hide irrelevant transform axes based on the functional light type
        if has_light:
            self.w_scl.setVisible(False) 
            if light_type == "Directional": self.w_pos.setVisible(False)
            elif light_type == "Point": self.w_rot.setVisible(False)

        self.fast_update(tf_data)

    def fast_update(self, tf_data: Dict[str, Any]) -> None:
        """Update all data (Used when an Entity is newly clicked)."""
        if not tf_data: return
        set_vec3_spinboxes(self.sp_pos, tf_data["pos"])
        set_vec3_spinboxes(self.sp_rot, tf_data["rot"])
        set_vec3_spinboxes(self.sp_scl, tf_data["scl"])

    def fast_update_single_axis(self, mode: str, values: tuple) -> None:
        """Super fast update of 1 axis (Used when dragging Gizmo with left mouse)."""
        if mode == "MOVE":
            set_vec3_spinboxes(self.sp_pos, values)
        elif mode == "ROTATE":
            set_vec3_spinboxes(self.sp_rot, values)
        elif mode == "SCALE":
            set_vec3_spinboxes(self.sp_scl, values)

    def reset_transform(self) -> None:
        if self._controller:
            self._controller.reset_transform()

    def apply_transform(self) -> None:
        """User changes values on UI -> Call Controller to save to Engine."""
        if not self._controller: return
        self._controller.set_property("Transform", "position", [s.value() for s in self.sp_pos])
        self._controller.set_property("Transform", "rotation", [s.value() for s in self.sp_rot])
        self._controller.set_property("Transform", "scale", [s.value() for s in self.sp_scl])