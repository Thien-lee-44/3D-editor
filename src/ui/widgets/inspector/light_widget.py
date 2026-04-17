import math
from typing import Any, Dict
from PySide6.QtWidgets import (QWidget, QHBoxLayout, QLabel, QCheckBox, 
                               QComboBox, QFormLayout, QVBoxLayout)

from src.ui.widgets.custom_inputs import SliderSpinBox
from src.ui.widgets.sun_hud_widget import SunHUDWidget
from .base_widget import BaseComponentWidget, set_vec3_spinboxes, rgb_to_hex
from src.app import AppEvent

from src.app.config import (
    DEFAULT_LIGHT_INTENSITY, DEFAULT_LIGHT_AMBIENT, DEFAULT_LIGHT_DIFFUSE, DEFAULT_LIGHT_SPECULAR,
    DEFAULT_LIGHT_CONSTANT, DEFAULT_LIGHT_LINEAR, DEFAULT_LIGHT_QUADRATIC,
    DEFAULT_SPOT_INNER_ANGLE, DEFAULT_SPOT_OUTER_ANGLE, DEFAULT_LIGHT_COLOR,
    LIGHT_INTENSITY_RANGE, LIGHT_FACTOR_RANGE, LIGHT_YAW_RANGE, LIGHT_PITCH_RANGE, 
    LIGHT_CUTOFF_RANGE, LIGHT_ATTEN_CONST_RANGE, LIGHT_ATTEN_LIN_RANGE, LIGHT_ATTEN_QUAD_RANGE
)

class LightWidget(BaseComponentWidget):
    def __init__(self, controller: Any) -> None:
        super().__init__("Light", controller)
        
        row_light_top = QHBoxLayout()
        self.chk_light_on = QCheckBox("Enable Light")
        self.chk_light_on.clicked.connect(self.request_undo_snapshot)
        self.chk_light_on.toggled.connect(self.apply_light)
        row_light_top.addWidget(self.chk_light_on)
        
        self.chk_light_proxy = QCheckBox("Show Proxy")
        self.chk_light_proxy.setMinimumWidth(150)
        self.chk_light_proxy.clicked.connect(self.request_undo_snapshot)
        self.chk_light_proxy.toggled.connect(self.apply_light_proxy)
        row_light_top.addWidget(self.chk_light_proxy)
        self.layout.addLayout(row_light_top)
        
        row_l_type = QHBoxLayout()
        row_l_type.addWidget(QLabel("Type:"))
        self.cmb_light_type = QComboBox()
        self.cmb_light_type.addItems(["Directional", "Point", "Spot"])
        self.cmb_light_type.setEnabled(False) 
        row_l_type.addWidget(self.cmb_light_type)
        self.layout.addLayout(row_l_type)

        flp = QFormLayout()
        self.sp_light_int = SliderSpinBox(*LIGHT_INTENSITY_RANGE, 0.1, DEFAULT_LIGHT_INTENSITY, self.apply_light, press_callback=self.request_undo_snapshot)
        flp.addRow("Intensity:", self.sp_light_int)
        self.layout.addLayout(flp)

        row_l_mode = QHBoxLayout()
        row_l_mode.addWidget(QLabel("Color Mode:"))
        self.cmb_light_mode = QComboBox()
        self.cmb_light_mode.addItems(["Basic (Base + Multipliers)", "Advanced (Independent)"])
        self.cmb_light_mode.activated.connect(self.request_undo_snapshot)
        self.cmb_light_mode.currentIndexChanged.connect(self.switch_light_mode)
        row_l_mode.addWidget(self.cmb_light_mode)
        self.layout.addLayout(row_l_mode)

        self.w_l_basic = QWidget()
        flb = QFormLayout(self.w_l_basic)
        flb.setContentsMargins(0, 0, 0, 0)
        row_base, self.btn_light_base, self.sp_light_base_vec = self._build_color_row('base', self.apply_light_vec_colors, self.pick_light_color)
        flb.addRow("Base Color:", row_base)
        self.sp_l_amb = SliderSpinBox(*LIGHT_FACTOR_RANGE, 0.05, DEFAULT_LIGHT_AMBIENT, self.apply_light, press_callback=self.request_undo_snapshot)
        self.sp_l_diff = SliderSpinBox(*LIGHT_FACTOR_RANGE, 0.05, DEFAULT_LIGHT_DIFFUSE, self.apply_light, press_callback=self.request_undo_snapshot)
        self.sp_l_spec = SliderSpinBox(*LIGHT_FACTOR_RANGE, 0.05, DEFAULT_LIGHT_SPECULAR, self.apply_light, press_callback=self.request_undo_snapshot)
        flb.addRow("Ambient Factor:", self.sp_l_amb)
        flb.addRow("Diffuse Factor:", self.sp_l_diff)
        flb.addRow("Specular Factor:", self.sp_l_spec)
        self.layout.addWidget(self.w_l_basic)

        self.w_l_adv = QWidget()
        fla = QFormLayout(self.w_l_adv)
        fla.setContentsMargins(0, 0, 0, 0)
        row_amb, self.btn_l_amb_c, self.sp_light_amb_vec = self._build_color_row('amb', self.apply_light_vec_colors, self.pick_light_color)
        row_diff, self.btn_l_diff_c, self.sp_light_diff_vec = self._build_color_row('diff', self.apply_light_vec_colors, self.pick_light_color)
        row_spec, self.btn_l_spec_c, self.sp_light_spec_vec = self._build_color_row('spec', self.apply_light_vec_colors, self.pick_light_color)
        fla.addRow("Ambient:", row_amb)
        fla.addRow("Diffuse:", row_diff)
        fla.addRow("Specular:", row_spec)
        self.layout.addWidget(self.w_l_adv)

        self.w_light_dir = QWidget()
        l_dir = QFormLayout(self.w_light_dir)
        l_dir.setContentsMargins(0, 0, 0, 0)
        self.sp_light_yaw = SliderSpinBox(*LIGHT_YAW_RANGE, 1.0, 0.0, self.apply_light_direction, press_callback=self.request_undo_snapshot)
        self.sp_light_pitch = SliderSpinBox(*LIGHT_PITCH_RANGE, 1.0, 0.0, self.apply_light_direction, press_callback=self.request_undo_snapshot)
        l_dir.addRow("Yaw:", self.sp_light_yaw)
        l_dir.addRow("Pitch:", self.sp_light_pitch)
        self.layout.addWidget(self.w_light_dir)

        self.w_light_spot = QWidget()
        l_spot = QFormLayout(self.w_light_spot)
        l_spot.setContentsMargins(0, 0, 0, 0)
        self.sp_light_cut = SliderSpinBox(*LIGHT_CUTOFF_RANGE, 0.5, DEFAULT_SPOT_INNER_ANGLE, self.apply_light, press_callback=self.request_undo_snapshot)
        self.sp_light_out = SliderSpinBox(*LIGHT_CUTOFF_RANGE, 0.5, DEFAULT_SPOT_OUTER_ANGLE, self.apply_light, press_callback=self.request_undo_snapshot)
        l_spot.addRow("Inner Cutoff:", self.sp_light_cut)
        l_spot.addRow("Outer Cutoff:", self.sp_light_out)
        self.layout.addWidget(self.w_light_spot)

        self.w_light_atten = QWidget()
        l_atten = QFormLayout(self.w_light_atten)
        l_atten.setContentsMargins(0, 0, 0, 0)
        self.sp_l_const = SliderSpinBox(*LIGHT_ATTEN_CONST_RANGE, 0.1, DEFAULT_LIGHT_CONSTANT, self.apply_light, press_callback=self.request_undo_snapshot)
        self.sp_l_lin = SliderSpinBox(*LIGHT_ATTEN_LIN_RANGE, 0.001, DEFAULT_LIGHT_LINEAR, self.apply_light, press_callback=self.request_undo_snapshot)
        self.sp_l_quad = SliderSpinBox(*LIGHT_ATTEN_QUAD_RANGE, 0.0001, DEFAULT_LIGHT_QUADRATIC, self.apply_light, press_callback=self.request_undo_snapshot)
        l_atten.addRow("Constant:", self.sp_l_const)
        l_atten.addRow("Linear:", self.sp_l_lin)
        l_atten.addRow("Quadratic:", self.sp_l_quad)
        self.layout.addWidget(self.w_light_atten)

        self.sun_hud_container = QWidget()
        sun_lay = QVBoxLayout(self.sun_hud_container)
        sun_lay.setContentsMargins(0, 10, 0, 0)
        sun_lay.addWidget(QLabel("Sun Direction Controller:"))
        
        self.sun_hud_widget = SunHUDWidget(self) 
        sun_lay.addWidget(self.sun_hud_widget)
        self.layout.addWidget(self.sun_hud_container)

    def update_data(self, ld: Dict[str, Any], mesh_visible: bool) -> None:
        self.cmb_light_type.blockSignals(True)
        self.cmb_light_type.setCurrentIndex(["Directional", "Point", "Spot"].index(ld["type"]) if ld.get("type") in ["Directional", "Point", "Spot"] else 1)
        self.cmb_light_type.blockSignals(False)
        
        self.w_light_dir.setVisible(ld.get("type") in ["Directional", "Spot"])
        self.w_light_spot.setVisible(ld.get("type") == "Spot")
        self.sun_hud_container.setVisible(ld.get("type") == "Directional")

        self.chk_light_on.blockSignals(True)
        self.chk_light_on.setChecked(ld.get("on", True))
        self.chk_light_on.blockSignals(False)
        
        self.chk_light_proxy.blockSignals(True)
        if ld.get("type") == "Directional":
            self.chk_light_proxy.setChecked(False)
            self.chk_light_proxy.setEnabled(False)
            self.chk_light_proxy.setText("Hidden (No Proxy)")
        else:
            self.chk_light_proxy.setEnabled(True)
            self.chk_light_proxy.setText("Show Proxy")
            self.chk_light_proxy.setChecked(mesh_visible)
        self.chk_light_proxy.blockSignals(False)
        
        self.sp_light_int.setValue(ld.get("intensity", DEFAULT_LIGHT_INTENSITY))
        
        self.cmb_light_mode.blockSignals(True)
        self.cmb_light_mode.setCurrentIndex(1 if ld.get("use_advanced_mode", False) else 0)
        self.cmb_light_mode.blockSignals(False)
        
        self.w_l_basic.setVisible(not ld.get("use_advanced_mode", False))
        self.w_l_adv.setVisible(ld.get("use_advanced_mode", False))

        self.sp_l_amb.setValue(ld.get("ambient_strength", DEFAULT_LIGHT_AMBIENT))
        self.sp_l_diff.setValue(ld.get("diffuse_strength", DEFAULT_LIGHT_DIFFUSE))
        self.sp_l_spec.setValue(ld.get("specular_strength", DEFAULT_LIGHT_SPECULAR))

        set_vec3_spinboxes(self.sp_light_base_vec, ld.get("color", list(DEFAULT_LIGHT_COLOR)))
        set_vec3_spinboxes(self.sp_light_amb_vec, ld.get("explicit_ambient", list(DEFAULT_LIGHT_COLOR)))
        set_vec3_spinboxes(self.sp_light_diff_vec, ld.get("explicit_diffuse", list(DEFAULT_LIGHT_COLOR)))
        set_vec3_spinboxes(self.sp_light_spec_vec, ld.get("explicit_specular", list(DEFAULT_LIGHT_COLOR)))

        self.btn_light_base.setStyleSheet(rgb_to_hex(ld.get("color", list(DEFAULT_LIGHT_COLOR))))
        self.btn_l_amb_c.setStyleSheet(rgb_to_hex(ld.get("explicit_ambient", list(DEFAULT_LIGHT_COLOR))))
        self.btn_l_diff_c.setStyleSheet(rgb_to_hex(ld.get("explicit_diffuse", list(DEFAULT_LIGHT_COLOR))))
        self.btn_l_spec_c.setStyleSheet(rgb_to_hex(ld.get("explicit_specular", list(DEFAULT_LIGHT_COLOR))))

        if ld.get("type") == "Spot":
            self.sp_light_cut.setValue(math.degrees(math.acos(max(-1.0, min(1.0, ld.get("cutOff", math.cos(math.radians(DEFAULT_SPOT_INNER_ANGLE))))))))
            self.sp_light_out.setValue(math.degrees(math.acos(max(-1.0, min(1.0, ld.get("outerCutOff", math.cos(math.radians(DEFAULT_SPOT_OUTER_ANGLE))))))))

        self.fast_update(ld)
        
        self.w_light_atten.setVisible(ld.get("type") in ["Point", "Spot"])
        self.sp_l_const.setValue(ld.get("constant", DEFAULT_LIGHT_CONSTANT))
        self.sp_l_lin.setValue(ld.get("linear", DEFAULT_LIGHT_LINEAR))
        self.sp_l_quad.setValue(ld.get("quadratic", DEFAULT_LIGHT_QUADRATIC))

    def fast_update(self, ld: Dict[str, Any]) -> None:
        if not ld or ld.get("type") not in ["Directional", "Spot"]: return
        self.sp_light_yaw.setValue(ld.get("yaw", 0.0))
        self.sp_light_pitch.setValue(ld.get("pitch", 0.0))

    def fast_update_rotation(self, rot_values: tuple) -> None:
        self.sp_light_pitch.blockSignals(True)
        self.sp_light_yaw.blockSignals(True)
        self.sp_light_pitch.setValue(rot_values[0]) 
        self.sp_light_yaw.setValue(rot_values[1])   
        self.sp_light_pitch.blockSignals(False)
        self.sp_light_yaw.blockSignals(False)

    def apply_light(self) -> None:
        if not self._controller: return
            
        self._controller.set_property("Light", "on", self.chk_light_on.isChecked())
        self._controller.set_property("Light", "intensity", self.sp_light_int.value())
        self._controller.set_property("Light", "ambient_strength", self.sp_l_amb.value())
        self._controller.set_property("Light", "diffuse_strength", self.sp_l_diff.value())
        self._controller.set_property("Light", "specular_strength", self.sp_l_spec.value())
        self._controller.set_property("Light", "constant", self.sp_l_const.value())
        self._controller.set_property("Light", "linear", self.sp_l_lin.value())
        self._controller.set_property("Light", "quadratic", self.sp_l_quad.value())
        
        if self.cmb_light_type.currentText() == "Spot":
            c, o = self.sp_light_cut.value(), self.sp_light_out.value()
            if c > o: 
                o = c
                self.sp_light_out.blockSignals(True)
                self.sp_light_out.setValue(o)
                self.sp_light_out.blockSignals(False)
            self._controller.set_property("Light", "cutOff", math.cos(math.radians(c)))
            self._controller.set_property("Light", "outerCutOff", math.cos(math.radians(o)))

    def apply_light_proxy(self) -> None:
        if not self._controller: return
        self._controller.set_property("Mesh", "visible", self.chk_light_proxy.isChecked())

    def switch_light_mode(self, idx: int) -> None:
        if not self._controller: return
        self._controller.set_property("Light", "use_advanced_mode", idx == 1)
        self.w_l_basic.setVisible(idx == 0)
        self.w_l_adv.setVisible(idx == 1)

    def apply_light_vec_colors(self) -> None:
        if not self._controller: return
        base_c = [s.value() for s in self.sp_light_base_vec]
        amb_c = [s.value() for s in self.sp_light_amb_vec]
        diff_c = [s.value() for s in self.sp_light_diff_vec]
        spec_c = [s.value() for s in self.sp_light_spec_vec]
        
        self._controller.set_property("Light", "color", base_c)
        self._controller.set_property("Light", "explicit_ambient", amb_c)
        self._controller.set_property("Light", "explicit_diffuse", diff_c)
        self._controller.set_property("Light", "explicit_specular", spec_c)
        
        self.btn_light_base.setStyleSheet(rgb_to_hex(base_c))
        self.btn_l_amb_c.setStyleSheet(rgb_to_hex(amb_c))
        self.btn_l_diff_c.setStyleSheet(rgb_to_hex(diff_c))
        self.btn_l_spec_c.setStyleSheet(rgb_to_hex(spec_c))

    def pick_light_color(self, c_type: str) -> None:
        from src.app import ctx
        data = ctx.engine.get_selected_entity_data()
        if not data or not data.get("light"): return
        
        prop_map = {
            'base': 'color', 
            'amb': 'explicit_ambient', 
            'diff': 'explicit_diffuse', 
            'spec': 'explicit_specular'
        }
        prop_name = prop_map.get(c_type)
        if not prop_name: return

        curr_c = data["light"].get(prop_name, list(DEFAULT_LIGHT_COLOR))
        new_c = self._pick_color_with_dialog(curr_c)
        
        if new_c is not None and self._controller:
            self.request_undo_snapshot()
            self._controller.set_property("Light", prop_name, new_c)
            
            vec_map = {
                'base': self.sp_light_base_vec, 
                'amb': self.sp_light_amb_vec, 
                'diff': self.sp_light_diff_vec, 
                'spec': self.sp_light_spec_vec
            }
            btn_map = {
                'base': self.btn_light_base, 
                'amb': self.btn_l_amb_c, 
                'diff': self.btn_l_diff_c, 
                'spec': self.btn_l_spec_c
            }
            
            set_vec3_spinboxes(vec_map[c_type], new_c)
            btn_map[c_type].setStyleSheet(rgb_to_hex(new_c))
            
            ctx.events.emit(AppEvent.SCENE_CHANGED)

    def apply_light_direction(self) -> None:
        if not self._controller: return
        self._controller.update_light_direction(self.sp_light_yaw.value(), self.sp_light_pitch.value())