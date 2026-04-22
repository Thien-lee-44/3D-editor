from typing import Any, Dict
from PySide6.QtWidgets import QFormLayout, QPushButton, QCheckBox, QComboBox, QLabel
from src.ui.widgets.custom_inputs import SliderSpinBox
from .base_widget import BaseComponentWidget

# Import SSOT configuration
from src.app.config import (
    DEFAULT_CAMERA_FOV, DEFAULT_CAMERA_NEAR, DEFAULT_CAMERA_FAR,
    CAMERA_FOV_RANGE, CAMERA_FOV_STEP, CAMERA_ORTHO_RANGE, 
    CAMERA_ORTHO_STEP, CAMERA_NEAR_RANGE, CAMERA_NEAR_STEP,
    CAMERA_FAR_RANGE, CAMERA_FAR_STEP,
    STYLE_BTN_ACTIVE_CAM, STYLE_BTN_INACTIVE_CAM
)

class CameraWidget(BaseComponentWidget):
    def __init__(self, controller: Any) -> None:
        super().__init__("Camera", controller)
        f_cam = QFormLayout()
        
        self.btn_set_cam_active = QPushButton("Set Active Camera")
        self.btn_set_cam_active.clicked.connect(self.set_active_camera)
        f_cam.addRow(self.btn_set_cam_active)
        
        self.chk_cam_proxy = QCheckBox("Show Proxy")
        self.chk_cam_proxy.clicked.connect(self.request_undo_snapshot)
        self.chk_cam_proxy.toggled.connect(self.apply_camera)
        f_cam.addRow("", self.chk_cam_proxy)
        
        self.cmb_cam_mode = QComboBox()
        self.cmb_cam_mode.addItems(["Perspective", "Orthographic"])
        self.cmb_cam_mode.activated.connect(self.request_undo_snapshot)
        self.cmb_cam_mode.currentIndexChanged.connect(self.apply_camera)
        f_cam.addRow("Projection:", self.cmb_cam_mode)
        
        self.lbl_cam_fov = QLabel("FOV:")
        self.sp_cam_fov = SliderSpinBox(*CAMERA_FOV_RANGE, CAMERA_FOV_STEP, DEFAULT_CAMERA_FOV, self.apply_camera, press_callback=self.request_undo_snapshot)
        f_cam.addRow(self.lbl_cam_fov, self.sp_cam_fov)

        self.lbl_cam_ortho = QLabel("Ortho Size:")
        self.sp_cam_ortho = SliderSpinBox(*CAMERA_ORTHO_RANGE, CAMERA_ORTHO_STEP, 5.0, self.apply_camera, press_callback=self.request_undo_snapshot)
        f_cam.addRow(self.lbl_cam_ortho, self.sp_cam_ortho)

        self.sp_cam_near = SliderSpinBox(*CAMERA_NEAR_RANGE, CAMERA_NEAR_STEP, DEFAULT_CAMERA_NEAR, self.apply_camera, press_callback=self.request_undo_snapshot)
        f_cam.addRow("Near Clip:", self.sp_cam_near)

        self.sp_cam_far = SliderSpinBox(*CAMERA_FAR_RANGE, CAMERA_FAR_STEP, DEFAULT_CAMERA_FAR, self.apply_camera, press_callback=self.request_undo_snapshot)
        f_cam.addRow("Far Clip:", self.sp_cam_far)

        self.layout.addLayout(f_cam)

    def update_data(self, cd: Dict[str, Any], mesh_visible: bool) -> None:
        is_active = bool(cd.get("is_active", cd.get("active", False)))
        if is_active:
            self.btn_set_cam_active.setText("Active Camera")
            self.btn_set_cam_active.setStyleSheet(STYLE_BTN_ACTIVE_CAM)
            self.btn_set_cam_active.setEnabled(False) 
            
            self.chk_cam_proxy.blockSignals(True)
            self.chk_cam_proxy.setChecked(False)
            self.chk_cam_proxy.setEnabled(False)
            self.chk_cam_proxy.setText("Hidden (Active View)")
            self.chk_cam_proxy.blockSignals(False)
        else:
            self.btn_set_cam_active.setText("Set Active Camera")
            self.btn_set_cam_active.setStyleSheet(STYLE_BTN_INACTIVE_CAM)
            self.btn_set_cam_active.setEnabled(True)
            
            self.chk_cam_proxy.blockSignals(True)
            self.chk_cam_proxy.setChecked(mesh_visible)
            self.chk_cam_proxy.setEnabled(True)
            self.chk_cam_proxy.setText("Show Proxy")
            self.chk_cam_proxy.blockSignals(False)

        self.cmb_cam_mode.blockSignals(True)
        self.sp_cam_fov.blockSignals(True)
        self.sp_cam_ortho.blockSignals(True)
        self.sp_cam_near.blockSignals(True)
        self.sp_cam_far.blockSignals(True)

        is_persp = (cd["mode"] == "Perspective")
        self.cmb_cam_mode.setCurrentIndex(0 if is_persp else 1)
        
        self.lbl_cam_fov.setVisible(is_persp)
        self.sp_cam_fov.setVisible(is_persp)
        self.lbl_cam_ortho.setVisible(not is_persp)
        self.sp_cam_ortho.setVisible(not is_persp)

        self.sp_cam_fov.setValue(cd.get("fov", DEFAULT_CAMERA_FOV))
        self.sp_cam_ortho.setValue(cd.get("ortho_size", cd.get("ortho", 5.0)))
        self.sp_cam_near.setValue(cd.get("near", DEFAULT_CAMERA_NEAR))
        self.sp_cam_far.setValue(cd.get("far", DEFAULT_CAMERA_FAR))

        self.cmb_cam_mode.blockSignals(False)
        self.sp_cam_fov.blockSignals(False)
        self.sp_cam_ortho.blockSignals(False)
        self.sp_cam_near.blockSignals(False)
        self.sp_cam_far.blockSignals(False)

    def set_active_camera(self) -> None:
        if not self._controller: return
        self.request_undo_snapshot()
        # This function calls ctx directly because the operation is closely related to the Engine
        from src.app import ctx, AppEvent
        ctx.engine.set_active_camera_selected()
        ctx.events.emit(AppEvent.ENTITY_SELECTED, ctx.engine.get_selected_entity_id())
        ctx.events.emit(AppEvent.SCENE_CHANGED)

    def apply_camera(self) -> None:
        if not self._controller: return
            
        mode = ["Perspective", "Orthographic"][self.cmb_cam_mode.currentIndex()]
        is_persp = (mode == "Perspective")

        self.lbl_cam_fov.setVisible(is_persp)
        self.sp_cam_fov.setVisible(is_persp)
        self.lbl_cam_ortho.setVisible(not is_persp)
        self.sp_cam_ortho.setVisible(not is_persp)

        self._controller.set_property("Camera", "mode", mode)
        self._controller.set_property("Camera", "fov", self.sp_cam_fov.value())
        self._controller.set_property("Camera", "ortho_size", self.sp_cam_ortho.value())
        self._controller.set_property("Camera", "near", self.sp_cam_near.value())
        self._controller.set_property("Camera", "far", self.sp_cam_far.value())

        if self.chk_cam_proxy.isEnabled(): 
            self._controller.set_property("Mesh", "visible", self.chk_cam_proxy.isChecked())
