from typing import Any, List, Optional
from PySide6.QtWidgets import QGroupBox, QVBoxLayout, QHBoxLayout, QPushButton, QColorDialog
from PySide6.QtGui import QColor

# Import from the new widgets directory
from src.ui.widgets.custom_inputs import create_vec3_input

# Import SSOT configuration
from src.app.config import (
    COLOR_VEC_RANGE, COLOR_VEC_STEP, 
    STYLE_COLOR_BTN_DARK_TEXT, STYLE_COLOR_BTN_LIGHT_TEXT
)

def rgb_to_hex(c_list: List[float]) -> str:
    """Converts a normalized RGB float array to a styled CSS string based on relative luminance."""
    r = max(0, min(255, int(c_list[0] * 255)))
    g = max(0, min(255, int(c_list[1] * 255)))
    b = max(0, min(255, int(c_list[2] * 255)))
    
    # Calculate relative luminance to determine optimal text contrast (Rec. 601)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    base_style = STYLE_COLOR_BTN_DARK_TEXT if lum > 128 else STYLE_COLOR_BTN_LIGHT_TEXT
    
    return f"background-color: rgb({r},{g},{b}); {base_style}"

def set_vec3_spinboxes(spinboxes: List[Any], values: List[float]) -> None:
    """Silently updates a list of spinboxes without triggering their 'valueChanged' signals."""
    for i in range(3):
        spinboxes[i].blockSignals(True)
        spinboxes[i].setValue(values[i])
        spinboxes[i].blockSignals(False)

class BaseComponentWidget(QGroupBox):
    """
    Parent class for all modular UI panels within the Inspector.
    Standardizes layout margins, title formatting, and color picking utility logic.
    """
    def __init__(self, title: str, controller: Any) -> None:
        super().__init__(title)
        self._controller = controller
        self.layout = QVBoxLayout(self)
        
        # Consistent vertical spacing across all inspector sub-panels
        self.layout.setContentsMargins(5, 10, 5, 5)
        self.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #555; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")

    def request_undo_snapshot(self) -> None:
        """Shared function to request the Controller to snapshot the scene before the user modifies values."""
        if self._controller and hasattr(self._controller, 'request_undo_snapshot'):
            self._controller.request_undo_snapshot()

    def _build_color_row(self, c_type: str, vec_callback: Any, btn_callback: Any) -> tuple:
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