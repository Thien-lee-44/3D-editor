"""
Math Generator View.

Provides the UI for the Procedural Math Surface Generator.
"""

from typing import Any, Tuple
from PySide6.QtWidgets import (QVBoxLayout, QHBoxLayout, QLabel, 
                               QLineEdit, QDoubleSpinBox, QSpinBox, QPushButton, QSlider)
from PySide6.QtCore import Qt

from src.ui.views.panels.base_panel import BasePanel
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

    def _build_axis_constraints(self, label_text: str) -> Tuple[QHBoxLayout, QDoubleSpinBox, QDoubleSpinBox]:
        """Helper to construct repetitive min/max spinbox layouts for axes."""
        row = QHBoxLayout()
        row.addWidget(QLabel(label_text))
        
        sp_min = QDoubleSpinBox()
        sp_min.setRange(MATH_LIMIT_MIN, MATH_LIMIT_MAX)
        sp_min.setValue(DEFAULT_MATH_RANGE[0])
        
        sp_max = QDoubleSpinBox()
        sp_max.setRange(MATH_LIMIT_MIN, MATH_LIMIT_MAX)
        sp_max.setValue(DEFAULT_MATH_RANGE[1])
        
        row.addWidget(sp_min)
        row.addWidget(QLabel("to"))
        row.addWidget(sp_max)
        
        return row, sp_min, sp_max

    def setup_ui(self) -> None:
        """Constructs the input forms and sliders."""
        self.layout = QVBoxLayout(self)
        
        self.layout.addWidget(QLabel("Function z = f(x,y):"))
        self.txt_func = QLineEdit(DEFAULT_MATH_FORMULA)
        self.layout.addWidget(self.txt_func)
        
        # --- Axis Constraints (Optimized with Helper) ---
        row_x, self.sp_x_min, self.sp_x_max = self._build_axis_constraints("X Axis:")
        self.layout.addLayout(row_x)
        
        row_y, self.sp_y_min, self.sp_y_max = self._build_axis_constraints("Y Axis:")
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
        """Links local UI events to controller triggers."""
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