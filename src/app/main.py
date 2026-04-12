import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QCoreApplication
from PySide6.QtGui import QSurfaceFormat
from src.app.config import APP_TITLE
# Absolute imports ensuring unified memory space
from src.engine import Engine
from src.app import ctx, AppEvent, config
from src.ui.controllers.main_ctrl import MainController

# IMPORT: Bring in the error handler initialization
from src.ui.error_handler import init_global_error_handler

def run_app() -> None:
    """
    Application Bootstrapper.
    Configures Qt/OpenGL systems, instantiates the 3D Engine backend, 
    and launches the Main Controller.
    """
    # 1. Global OpenGL Surface Configuration
    fmt = QSurfaceFormat()
    fmt.setSamples(config.MSAA_SAMPLES) 
    QSurfaceFormat.setDefaultFormat(fmt)
    
    QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    app.setStyle("Fusion")
    
    # =========================================================================
    # 2. INITIALIZE ERROR HANDLER (Must be exactly here: after app creation)
    # =========================================================================
    init_global_error_handler()
    
    try:
        # 3. CORE BOOTSTRAP
        engine_instance = Engine()
        ctx.engine = engine_instance
        
        # 4. UI BOOTSTRAP
        root_controller = MainController()
        root_controller.main_window.show()
        
        # 5. ASSET AUTOLOAD
        ctx.engine.auto_load_default_assets(config.TEXTURES_DIR)
        
        ctx.events.emit(AppEvent.HIERARCHY_NEEDS_REFRESH)     
        ctx.events.emit(AppEvent.ASSET_BROWSER_NEEDS_REFRESH) 
        
        current_id = ctx.engine.get_selected_entity_id()
        ctx.events.emit(AppEvent.ENTITY_SELECTED, current_id)
        ctx.events.emit(AppEvent.SCENE_CHANGED)               
        
        # 6. ENTER QT EVENT LOOP
        sys.exit(app.exec())
        
    except Exception as e:
        print(e)
        sys.exit(1)