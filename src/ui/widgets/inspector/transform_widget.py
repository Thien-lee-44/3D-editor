"""
Transform Widget.

Provides the Inspector UI for configuring an entity's 3D spatial properties
(Position, Rotation, Scale). Maintains extreme precision using internal shadow states.
"""

from typing import Any, Dict, List, Tuple
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
    """
    Inspector widget handling exact spatial values.
    Implements a 64-bit shadow state to prevent floating-point precision loss 
    caused by UI truncation when evaluating non-uniform hierarchical scaling.
    """

    def __init__(self, controller: Any) -> None:
        super().__init__("Transform", controller)
        self._controller = controller 
        
        self.btn_reset = QPushButton("Reset Transform")
        self.btn_reset.setStyleSheet(STYLE_BTN_RESET)
        self.btn_reset.clicked.connect(self.reset_transform)
        self.layout.addWidget(self.btn_reset)
        
        self.w_pos, self.sp_pos = create_vec3_input("Position:", self.apply_position, step=TRANSFORM_POS_STEP, press_callback=self.request_undo)
        self.w_rot, self.sp_rot = create_vec3_input("Rotation:", self.apply_rotation, min_val=TRANSFORM_ROT_RANGE[0], max_val=TRANSFORM_ROT_RANGE[1], step=TRANSFORM_ROT_STEP, press_callback=self.request_undo)
        self.w_scl, self.sp_scl = create_vec3_input("Scale:", self.apply_scale, default=DEFAULT_SPAWN_SCALE[0], min_val=TRANSFORM_SCL_MIN, step=TRANSFORM_SCL_STEP, press_callback=self.request_undo)
        
        self.layout.addWidget(self.w_pos)
        self.layout.addWidget(self.w_rot)
        self.layout.addWidget(self.w_scl)

        self._exact_pos: List[float] = [0.0, 0.0, 0.0]
        self._exact_rot: List[float] = [0.0, 0.0, 0.0]
        self._exact_scl: List[float] = [1.0, 1.0, 1.0]

    def request_undo(self) -> None:
        """Triggers an Undo snapshot before an incoming modification."""
        if self._controller:
            self._controller.request_undo_snapshot()

    def update_data(self, tf_data: Dict[str, Any]) -> None:
        """Refreshes the UI inputs taking into account active lock states (e.g., rigidbodies)."""
        locked = tf_data.get("locked_axes", {"pos": False, "rot": False, "scl": False})

        self.w_pos.setEnabled(not locked.get("pos", False))
        self.w_rot.setEnabled(not locked.get("rot", False))
        self.w_scl.setEnabled(not locked.get("scl", False))

        self.fast_update(tf_data)

    def fast_update(self, tf_data: Dict[str, Any]) -> None:
        """Updates the physical input displays silently without firing cascading UI events."""
        if not tf_data: 
            return
            
        self._exact_pos = list(tf_data.get("position", [0.0, 0.0, 0.0]))
        self._exact_rot = list(tf_data.get("rotation", [0.0, 0.0, 0.0]))
        self._exact_scl = list(tf_data.get("scale", [1.0, 1.0, 1.0]))

        self._block_all_signals(True)
        set_vec3_spinboxes(self.sp_pos, self._exact_pos)
        set_vec3_spinboxes(self.sp_rot, self._exact_rot)
        set_vec3_spinboxes(self.sp_scl, self._exact_scl)
        self._block_all_signals(False)

    def fast_update_single_axis(self, mode: str, values: Tuple[float, float, float]) -> None:
        """Rapidly updates a specific spatial vector, commonly used during Viewport gizmo dragging."""
        self._block_all_signals(True)
        if mode == "MOVE" and self.w_pos.isEnabled():
            self._exact_pos = list(values)
            set_vec3_spinboxes(self.sp_pos, values)
        elif mode == "ROTATE" and self.w_rot.isEnabled():
            self._exact_rot = list(values)
            set_vec3_spinboxes(self.sp_rot, values)
        elif mode == "SCALE" and self.w_scl.isEnabled():
            self._exact_scl = list(values)
            set_vec3_spinboxes(self.sp_scl, values)
        self._block_all_signals(False)

    def _block_all_signals(self, state: bool) -> None:
        """Bulk utility to block/unblock signals for all transform input boxes."""
        for sp in self.sp_pos + self.sp_rot + self.sp_scl:
            sp.blockSignals(state)

    def reset_transform(self) -> None:
        """Commands the controller to zero out all transformations for this entity."""
        if self._controller:
            self._controller.reset_transform()

    def apply_position(self) -> None:
        """Commits the local position modifications to the backend component."""
        if not self._controller or not self.w_pos.isEnabled(): 
            return
            
        sender = self.sender()
        updated = False
        for i, sp in enumerate(self.sp_pos):
            # Prioritize retrieving exact values if manually typed by the user
            if sender == sp:
                self._exact_pos[i] = sp.value()
                updated = True
        
        # Safe fallback: Check differences if signal origin is ambiguous
        if not updated:
            for i, sp in enumerate(self.sp_pos):
                ui_val = sp.value()
                if abs(ui_val - round(self._exact_pos[i], sp.decimals())) > 1e-5:
                    self._exact_pos[i] = ui_val
                    
        self._controller.set_property("Transform", "position", self._exact_pos)

    def apply_rotation(self) -> None:
        """Commits the local Euler rotation modifications to the backend component."""
        if not self._controller or not self.w_rot.isEnabled(): 
            return
            
        sender = self.sender()
        updated = False
        for i, sp in enumerate(self.sp_rot):
            if sender == sp:
                self._exact_rot[i] = sp.value()
                updated = True
        
        if not updated:
            for i, sp in enumerate(self.sp_rot):
                ui_val = sp.value()
                if abs(ui_val - round(self._exact_rot[i], sp.decimals())) > 1e-5:
                    self._exact_rot[i] = ui_val
                    
        self._controller.set_property("Transform", "rotation", self._exact_rot)

    def apply_scale(self) -> None:
        """Commits the local scale modifications to the backend component."""
        if not self._controller or not self.w_scl.isEnabled(): 
            return
            
        sender = self.sender()
        updated = False
        for i, sp in enumerate(self.sp_scl):
            if sender == sp:
                self._exact_scl[i] = sp.value()
                updated = True
        
        if not updated:
            for i, sp in enumerate(self.sp_scl):
                ui_val = sp.value()
                if abs(ui_val - round(self._exact_scl[i], sp.decimals())) > 1e-5:
                    self._exact_scl[i] = ui_val
                    
        self._controller.set_property("Transform", "scale", self._exact_scl)