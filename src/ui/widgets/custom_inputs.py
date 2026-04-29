"""
Custom Input Widgets.

Provides specialized Qt input controls like tracked spinboxes and composite 
slider-spinboxes designed to integrate safely with the Undo/Redo history system.
"""

from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QDoubleSpinBox, QSlider
from PySide6.QtCore import Qt, Signal
from typing import Callable, Optional, Tuple, List, Any
from PySide6.QtGui import QFocusEvent, QMouseEvent, QKeyEvent

from src.app.config import GLOBAL_NUMERIC_MIN, GLOBAL_NUMERIC_MAX


class TrackedSpinBox(QDoubleSpinBox):
    """
    A custom spinbox designed to interact safely with the Undo/Redo system.
    Standard spinboxes emit 'valueChanged' on every keystroke (e.g., typing '1', '2', '0' 
    creates 3 separate states). This class intercepts focus and click events to emit 
    an 'editingStarted' signal exactly once before the value mutates, allowing the 
    ProjectManager to record a single, clean snapshot of the pre-edit state.
    """
    editingStarted = Signal()
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._has_emitted: bool = False

    def focusInEvent(self, event: QFocusEvent) -> None:
        """Triggers a snapshot when the user tabs into or clicks the input field."""
        if not self._has_emitted:
            self.editingStarted.emit()
            self._has_emitted = True
        super().focusInEvent(event)

    def focusOutEvent(self, event: QFocusEvent) -> None:
        """Resets the emission flag when the user finishes editing."""
        self._has_emitted = False
        super().focusOutEvent(event)
        
    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Ensures snapshot emission if the field is clicked without prior focus."""
        if not self.hasFocus() and not self._has_emitted:
            self.editingStarted.emit()
            self._has_emitted = True
        super().mousePressEvent(event)
        
    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Ensures snapshot emission upon initial keystrokes."""
        if not self._has_emitted:
            self.editingStarted.emit()
            self._has_emitted = True
        super().keyPressEvent(event)


class SliderSpinBox(QWidget):
    """
    A composite UI control combining a horizontal slider with a numerical spinbox.
    Requires careful signal management to prevent recursive feedback loops.
    """
    
    def __init__(self, min_val: float, max_val: float, step: float, default_val: float, callback: Callable[[], None], press_callback: Optional[Callable[[], None]] = None) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.spin = TrackedSpinBox()
        self.spin.setRange(min_val, max_val)
        self.spin.setSingleStep(step)
        self.spin.setValue(default_val)
        
        self.slider = QSlider(Qt.Horizontal)
        # QSlider only supports integers. We apply a multiplier to simulate floating-point precision.
        self.multiplier: float = 100.0 if (max_val - min_val) <= 10.0 else 1.0
        self.slider.setRange(int(min_val * self.multiplier), int(max_val * self.multiplier))
        self.slider.setValue(int(default_val * self.multiplier))
        
        self.callback = callback
        self.press_callback = press_callback
        
        # Cross-connect the widgets so updating one updates the other
        self.spin.valueChanged.connect(self._sync_spin)
        self.slider.valueChanged.connect(self._sync_slider)
        
        if self.press_callback:
            self.slider.sliderPressed.connect(self.press_callback)
            self.spin.editingStarted.connect(self.press_callback)
        
        layout.addWidget(self.slider)
        layout.addWidget(self.spin)
        
    def _sync_spin(self, val: float) -> None:
        """Called when the SpinBox changes. Updates the Slider while blocking its signals to prevent a feedback loop."""
        self.slider.blockSignals(True)
        self.slider.setValue(int(val * self.multiplier))
        self.slider.blockSignals(False)
        if self.callback: 
            self.callback()
        
    def _sync_slider(self, val: int) -> None:
        """Called when the Slider moves. Updates the SpinBox while blocking its signals."""
        self.spin.blockSignals(True)
        self.spin.setValue(val / self.multiplier)
        self.spin.blockSignals(False)
        if self.callback: 
            self.callback()
        
    def value(self) -> float: 
        """Returns the current floating-point value from the spinbox."""
        return self.spin.value()
    
    def setValue(self, val: float) -> None:
        """Programmatically sets the value of both sub-widgets safely."""
        self.spin.setValue(val)
        self.slider.blockSignals(True)
        self.slider.setValue(int(val * self.multiplier))
        self.slider.blockSignals(False)
        
    def blockSignals(self, b: bool) -> bool:
        """Overrides base method to ensure both child widgets have their signals blocked simultaneously."""
        self.spin.blockSignals(b)
        self.slider.blockSignals(b)
        return super().blockSignals(b)


def create_vec3_input(label_text: str, callback: Callable[[], None], default: float = 0.0, min_val: float = GLOBAL_NUMERIC_MIN, max_val: float = GLOBAL_NUMERIC_MAX, step: float = 0.1, press_callback: Optional[Callable[[], None]] = None) -> Tuple[QWidget, List[TrackedSpinBox]]:
    """
    Factory function to quickly generate a labeled, 3-component (X, Y, Z) vector input row.
    Returns the container widget and a list referencing the three TrackedSpinBox instances.
    """
    w = QWidget()
    row = QHBoxLayout(w)
    row.setContentsMargins(0, 0, 0, 0)
    
    lbl = QLabel(label_text)
    lbl.setMinimumWidth(50)
    row.addWidget(lbl)
    
    spins: List[TrackedSpinBox] = []
    for _ in range(3):
        s = TrackedSpinBox()
        s.setRange(min_val, max_val)
        s.setSingleStep(step)
        s.setValue(default)
        s.valueChanged.connect(callback)
        
        if press_callback:
            s.editingStarted.connect(press_callback)
            
        row.addWidget(s)
        spins.append(s)
        
    return w, spins