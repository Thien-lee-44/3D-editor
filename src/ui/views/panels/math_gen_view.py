from PySide6.QtWidgets import (QVBoxLayout, QHBoxLayout, QLabel, 
                               QLineEdit, QDoubleSpinBox, QSpinBox, QPushButton, QSlider)
from PySide6.QtCore import Qt
from typing import Any

from src.ui.views.panels.base_panel import BasePanel

# Import SSOT configuration
from src.app.config import (
    PANEL_TITLE_MATH_GEN, DEFAULT_MATH_FORMULA, DEFAULT_MATH_RANGE, DEFAULT_MATH_RESOLUTION,
    MATH_LIMIT_MIN, MATH_LIMIT_MAX, MATH_RES_MIN, MATH_RES_MAX
)

class MathGeneratorPanelView(BasePanel):
    """
    Dumb View for the Procedural Math Surface Generator.
    Collects UI parameters and dispatches them to the MathGenController.
    """
    PANEL_TITLE = PANEL_TITLE_MATH_GEN
    PANEL_DOCK_AREA = Qt.BottomDockWidgetArea

    def setup_ui(self) -> None:
        self.layout = QVBoxLayout(self)
        
        self.layout.addWidget(QLabel("Function z = f(x,y):"))
        self.txt_func = QLineEdit(DEFAULT_MATH_FORMULA)
        self.layout.addWidget(self.txt_func)
        
        # --- X Axis Constraints ---
        row_x = QHBoxLayout()
        row_x.addWidget(QLabel("X Axis:"))
        self.sp_x_min = QDoubleSpinBox()
        self.sp_x_min.setRange(MATH_LIMIT_MIN, MATH_LIMIT_MAX)
        self.sp_x_min.setValue(DEFAULT_MATH_RANGE[0])
        
        self.sp_x_max = QDoubleSpinBox()
        self.sp_x_max.setRange(MATH_LIMIT_MIN, MATH_LIMIT_MAX)
        self.sp_x_max.setValue(DEFAULT_MATH_RANGE[1])
        
        row_x.addWidget(self.sp_x_min)
        row_x.addWidget(QLabel("to"))
        row_x.addWidget(self.sp_x_max)
        self.layout.addLayout(row_x)

        # --- Y Axis Constraints ---
        row_y = QHBoxLayout()
        row_y.addWidget(QLabel("Y Axis:"))
        self.sp_y_min = QDoubleSpinBox()
        self.sp_y_min.setRange(MATH_LIMIT_MIN, MATH_LIMIT_MAX)
        self.sp_y_min.setValue(DEFAULT_MATH_RANGE[0])
        
        self.sp_y_max = QDoubleSpinBox()
        self.sp_y_max.setRange(MATH_LIMIT_MIN, MATH_LIMIT_MAX)
        self.sp_y_max.setValue(DEFAULT_MATH_RANGE[1])
        
        row_y.addWidget(self.sp_y_min)
        row_y.addWidget(QLabel("to"))
        row_y.addWidget(self.sp_y_max)
        self.layout.addLayout(row_y)

        # --- Grid Resolution ---
        self.layout.addWidget(QLabel("Grid Resolution:"))
        row_res = QHBoxLayout()
        
        self.slider_res = QSlider(Qt.Horizontal)
        self.slider_res.setRange(MATH_RES_MIN, MATH_RES_MAX)
        self.slider_res.setValue(DEFAULT_MATH_RESOLUTION)
        
        self.sp_res = QSpinBox()
        self.sp_res.setRange(MATH_RES_MIN, MATH_RES_MAX)
        self.sp_res.setValue(DEFAULT_MATH_RESOLUTION)
        
        row_res.addWidget(self.slider_res)
        row_res.addWidget(self.sp_res)
        self.layout.addLayout(row_res)
        
        self.btn_gen = QPushButton("Generate 3D Surface")
        self.layout.addWidget(self.btn_gen)
        self.layout.addStretch(1)

    def bind_events(self) -> None:
        self.slider_res.valueChanged.connect(self.sp_res.setValue)
        self.sp_res.valueChanged.connect(self.slider_res.setValue)
        self.btn_gen.clicked.connect(self._on_generate)

    def _on_generate(self) -> None:
        if self._controller:
            self._controller.generate_surface(
                self.txt_func.text(),
                self.sp_x_min.value(),
                self.sp_x_max.value(),
                self.sp_y_min.value(),
                self.sp_y_max.value(),
                self.sp_res.value()
            )