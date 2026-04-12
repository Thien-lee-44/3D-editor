from PySide6.QtWidgets import QWidget
from typing import Any, Optional

class BaseView(QWidget):
    """
    The foundational class for all Views in the MVC architecture.
    Views are 'dumb'; they do not contain engine logic. They only render UI 
    and report user interactions back to their Controller.
    """
    def __init__(self, controller: Optional[Any] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._controller = controller
        
        self.setup_ui()
        self.bind_events()

    def setup_ui(self) -> None:
        """Defines layout and instantiates child widgets. Must be overridden."""
        pass

    def bind_events(self) -> None:
        """Connects Qt signals (clicks, text changes) to Controller methods. Must be overridden."""
        pass