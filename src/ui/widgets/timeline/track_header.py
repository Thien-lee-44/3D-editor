"""
Track Header Widget.

Displays the hierarchy and metadata of the currently animated entity.
Locked to the left of the timeline interface.
"""

from typing import Any
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtGui import QColor, QPainter
from PySide6.QtCore import Qt


class TrackHeaderWidget(QWidget):
    """
    Displays the hierarchy and names of the animated properties.
    Positioned on the left side of the timeline panel, locked in sync with the Dope Sheet.
    """
    
    def __init__(self) -> None:
        super().__init__()
        self.setFixedWidth(180)
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.layout.setSpacing(0)
        
        # Entity Target Label
        self.lbl_target = QLabel("No Entity Selected")
        self.lbl_target.setStyleSheet("color: #FFA500; font-weight: bold; font-size: 11px;")
        
        # Track Name Label
        self.lbl_track = QLabel(" Master Animation")
        self.lbl_track.setStyleSheet("color: #CCCCCC; font-size: 11px;")
        
        self.layout.addWidget(self.lbl_target)
        self.layout.addWidget(self.lbl_track)
        self.layout.addStretch()

    def paintEvent(self, event: Any) -> None:
        """Renders the fixed dark background for the header area."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(35, 35, 35))
        
    def set_target_name(self, name: str) -> None:
        """Updates the entity name label dynamically."""
        self.lbl_target.setText(name if name else "No Entity Selected")