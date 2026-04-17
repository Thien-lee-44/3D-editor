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
        super().__init__("Transform", controller)
        self._controller = controller 
        
        self.btn_reset = QPushButton("Reset Transform")
        self.btn_reset.setStyleSheet(STYLE_BTN_RESET)
        self.btn_reset.clicked.connect(self.reset_transform)
        self.layout.addWidget(self.btn_reset)
        
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

    def update_data(self, tf_data: Dict[str, Any]) -> None:
        """Applies dynamic constraints and strictly maps data to the UI."""
        locked = tf_data.get("locked_axes", {"pos": False, "rot": False, "scl": False})

        # By disabling the wrapper widgets, Qt gracefully grays out all internal labels and spinboxes,
        # preventing user edits while maintaining layout stability (no jittering).
        self.w_pos.setEnabled(not locked.get("pos", False))
        self.w_rot.setEnabled(not locked.get("rot", False))
        self.w_scl.setEnabled(not locked.get("scl", False))

        self.fast_update(tf_data)

    def fast_update(self, tf_data: Dict[str, Any]) -> None:
        """Update all data mapping exactly to JSON property names."""
        if not tf_data: return
        set_vec3_spinboxes(self.sp_pos, tf_data.get("position", [0, 0, 0]))
        set_vec3_spinboxes(self.sp_rot, tf_data.get("rotation", [0, 0, 0]))
        set_vec3_spinboxes(self.sp_scl, tf_data.get("scale", [1, 1, 1]))

    def fast_update_single_axis(self, mode: str, values: tuple) -> None:
        """Super fast update of 1 axis (Used when dragging Gizmo with left mouse)."""
        if mode == "MOVE" and self.w_pos.isEnabled():
            set_vec3_spinboxes(self.sp_pos, values)
        elif mode == "ROTATE" and self.w_rot.isEnabled():
            set_vec3_spinboxes(self.sp_rot, values)
        elif mode == "SCALE" and self.w_scl.isEnabled():
            set_vec3_spinboxes(self.sp_scl, values)

    def reset_transform(self) -> None:
        if self._controller:
            self._controller.reset_transform()

    def apply_transform(self) -> None:
        """Sends exactly the property names expected by the Backend."""
        if not self._controller: return
        
        # Only commit changes if the axis is unlocked
        if self.w_pos.isEnabled():
            self._controller.set_property("Transform", "position", [s.value() for s in self.sp_pos])
        if self.w_rot.isEnabled():
            self._controller.set_property("Transform", "rotation", [s.value() for s in self.sp_rot])
        if self.w_scl.isEnabled():
            self._controller.set_property("Transform", "scale", [s.value() for s in self.sp_scl])