"""
Centralized UI Error Handler.

Captures exceptions from any layer or thread and safely routes them 
to the main UI thread via Qt Signals to display a QMessageBox.
"""

import logging
import functools
from typing import Callable, Any, Optional

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QMessageBox, QApplication

# IMPORT: Domain Exceptions to filter expected vs unexpected errors
from src.app.exceptions import EngineError


class ErrorDispatcher(QObject):
    """
    Thread-safe error dispatcher. 
    Listens for errors and triggers UI popups exclusively on the main thread.
    """
    
    # Signal signature: (context_title, error_message)
    error_occurred = Signal(str, str)

    def __init__(self) -> None:
        super().__init__()
        self.error_occurred.connect(self._show_error_dialog)

    def handle(self, exception: Exception, context: str = "Application Error") -> None:
        """
        Entry point for caught exceptions. 
        Smart logging: Filters out tracebacks for expected domain constraints.
        """
        # Classify the error type
        if isinstance(exception, EngineError):
            # Expected domain rule violation (e.g., SimulationError, ResourceError)
            # Log cleanly without stack trace
            logging.warning(f"[{context}] Engine Constraint Violated: {exception}")
        else:
            # Unexpected Python runtime crash (e.g., TypeError, KeyError)
            # Log full stack trace for debugging
            logging.error(f"[{context}] Unexpected Exception: {exception}", exc_info=True)

        # Safely emit signal to cross thread boundaries if necessary
        self.error_occurred.emit(context, str(exception))

    def _show_error_dialog(self, title: str, message: str) -> None:
        """
        Slot executed to display the actual popup dialog.
        Strictly executed on the main UI thread managed by Qt's event loop.
        """
        if not QApplication.instance():
            return
            
        dialog = QMessageBox()
        dialog.setIcon(QMessageBox.Warning)
        dialog.setWindowTitle(title)
        dialog.setText("Operation Rejected:")
        dialog.setInformativeText(message)
        dialog.exec()


# Global singleton instance, strictly initialized AFTER QApplication
global_error_handler: Optional[ErrorDispatcher] = None


def init_global_error_handler() -> None:
    """Instantiates the dispatcher once the Qt Environment is ready."""
    global global_error_handler
    if global_error_handler is None:
        global_error_handler = ErrorDispatcher()


def safe_execute(context: str = "Execution Error") -> Callable:
    """
    Decorator to wrap UI callbacks. Catches any unhandled exceptions 
    and forwards them to the centralized global error handler.
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                if global_error_handler:
                    global_error_handler.handle(e, context)
                else:
                    logging.error(f"[{context}] Dispatcher offline. Error: {e}", exc_info=True)
        return wrapper
    return decorator