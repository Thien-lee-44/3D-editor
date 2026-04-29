"""
Application Bootstrapper.
Initializes context, UI frameworks, and underlying 3D subsystems.
"""

import sys
import logging
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QCoreApplication
from PySide6.QtGui import QSurfaceFormat
from src.app.config import APP_TITLE, MSAA_SAMPLES, TEXTURES_DIR
from src.engine import Engine
from src.app import ctx, AppEvent
from src.ui.controllers.main_ctrl import MainController
from src.ui.error_handler import init_global_error_handler

def run_app() -> None:
    """
    Configures Qt and OpenGL systems, provisions the Engine instance,
    and executes the primary UI event loop.
    """
    # Configure global OpenGL format
    fmt = QSurfaceFormat()
    fmt.setSamples(MSAA_SAMPLES) 
    QSurfaceFormat.setDefaultFormat(fmt)
    
    QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    app.setStyle("Fusion")
    
    # Bind custom error handling post-QApplication initialization
    init_global_error_handler()
    
    try:
        # Initialize Core Engine and inject into Singleton Context
        engine_instance = Engine()
        ctx.engine = engine_instance
        
        # Initialize and display Main UI
        root_controller = MainController()
        root_controller.main_window.show()
        
        # Load default system resources
        ctx.engine.auto_load_default_assets(TEXTURES_DIR)
        
        # Broadcast initial state requirements
        ctx.events.emit(AppEvent.HIERARCHY_NEEDS_REFRESH)     
        ctx.events.emit(AppEvent.ASSET_BROWSER_NEEDS_REFRESH) 
        
        current_id = ctx.engine.get_selected_entity_id()
        ctx.events.emit(AppEvent.ENTITY_SELECTED, current_id)
        ctx.events.emit(AppEvent.SCENE_CHANGED)               
        
        # Launch UI event loop
        sys.exit(app.exec())
        
    except Exception as e:
        logging.critical(f"Fatal application error during bootstrap: {e}", exc_info=True)
        sys.exit(1)