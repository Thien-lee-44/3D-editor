"""
Base Component Widget.

Provides the foundational UI class and shared utilities for all modular panels 
within the Inspector. Includes an accordion-style collapsible layout.
"""

from typing import Any, List, Optional, Tuple
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                               QColorDialog, QFrame)
from PySide6.QtGui import QColor
from PySide6.QtCore import Qt

from src.ui.widgets.custom_inputs import create_vec3_input
from src.app.config import (
    COLOR_VEC_RANGE, COLOR_VEC_STEP, 
    STYLE_COLOR_BTN_DARK_TEXT, STYLE_COLOR_BTN_LIGHT_TEXT
)


def rgb_to_hex(c_list: List[float]) -> str:
    """Converts a normalized RGB float array to a styled CSS string based on relative luminance."""
    r = max(0, min(255, int(c_list[0] * 255)))
    g = max(0, min(255, int(c_list[1] * 255)))
    b = max(0, min(255, int(c_list[2] * 255)))
    
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    base_style = STYLE_COLOR_BTN_DARK_TEXT if lum > 128 else STYLE_COLOR_BTN_LIGHT_TEXT
    
    return f"background-color: rgb({r},{g},{b}); {base_style}"


def set_vec3_spinboxes(spinboxes: List[Any], values: List[float]) -> None:
    """Silently updates a list of spinboxes without triggering their 'valueChanged' signals."""
    for i in range(3):
        spinboxes[i].blockSignals(True)
        spinboxes[i].setValue(values[i])
        spinboxes[i].blockSignals(False)


class BaseComponentWidget(QWidget):
    """
    Parent class for all modular UI panels within the Inspector.
    Implements a collapsible accordion architecture to optimize vertical screen space.
    Subclasses seamlessly inject their specific forms into the internal 'self.layout'.
    """
    
    def __init__(self, title: str, controller: Any) -> None:
        super().__init__()
        self._controller = controller
        self._is_collapsed: bool = False
        self._title_text: str = title

        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.setSpacing(0)

        # 1. Collapsible Header Button
        self.btn_toggle = QPushButton()
        self.btn_toggle.setCursor(Qt.PointingHandCursor)
        self.btn_toggle.setStyleSheet("""
            QPushButton {
                background-color: #333333;
                color: #E0E0E0;
                border: none;
                border-radius: 4px;
                padding: 6px 10px;
                text-align: left;
                font-weight: bold;
                font-size: 12px;
                margin-top: 6px;
            }
            QPushButton:hover {
                background-color: #444444;
            }
        """)
        self._update_button_text()
        self.btn_toggle.clicked.connect(self._on_toggle)
        self._main_layout.addWidget(self.btn_toggle)

        # 2. Content Container (Wrapped in QFrame to allow hiding)
        self.content_container = QFrame()
        
        # 3. Public Layout (Child widgets will automatically inject here)
        self.layout = QVBoxLayout(self.content_container)
        self.layout.setContentsMargins(10, 8, 0, 8) 
        self.layout.setSpacing(4)

        self._main_layout.addWidget(self.content_container)

    def _update_button_text(self) -> None:
        """Updates the visual indicator based on the current collapse state."""
        arrow = "►" if self._is_collapsed else "▼"
        self.btn_toggle.setText(f"{arrow}  {self._title_text}")

    def _on_toggle(self) -> None:
        """Handles the state mutation and triggers layout recalculation."""
        self._is_collapsed = not self._is_collapsed
        self.content_container.setVisible(not self._is_collapsed)
        self._update_button_text()

    # =========================================================================
    # LEGACY UTILITIES (Maintained for child class backward compatibility)
    # =========================================================================

    def request_undo_snapshot(self) -> None:
        """Shared function to request the Controller to snapshot the scene before value modification."""
        if self._controller and hasattr(self._controller, 'request_undo_snapshot'):
            self._controller.request_undo_snapshot()

    def _build_color_row(self, c_type: str, vec_callback: Any, btn_callback: Any) -> Tuple[QHBoxLayout, QPushButton, List[Any]]:
        """Assembles a standardized row containing a Color Dialog Button paired with explicit RGB spinboxes."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        
        btn = QPushButton("Color")
        btn.setFixedSize(45, 24)
        btn.clicked.connect(lambda: btn_callback(c_type))
        row.addWidget(btn)
        
        w_vec, sp_vec = create_vec3_input("", vec_callback, min_val=COLOR_VEC_RANGE[0], max_val=COLOR_VEC_RANGE[1], step=COLOR_VEC_STEP, press_callback=self.request_undo_snapshot)
        row.addWidget(w_vec)
        return row, btn, sp_vec

    def _pick_color_with_dialog(self, current_c_list: List[float]) -> Optional[List[float]]:
        """Utility function to open the OS-native Qt color picker dialog."""
        r, g, b = [max(0, min(255, int(c * 255))) for c in current_c_list[:3]]
        dialog = QColorDialog(self)
        dialog.setCurrentColor(QColor(r, g, b))
        
        if dialog.exec() == QColorDialog.Accepted:
            color = dialog.currentColor()
            return [color.red() / 255.0, color.green() / 255.0, color.blue() / 255.0]
        return None