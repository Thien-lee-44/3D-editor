from PySide6.QtWidgets import QProgressDialog, QApplication
from PySide6.QtCore import Qt, QEventLoop

from src.app import ctx, AppEvent
from src.app.exceptions import SimulationError
from src.ui.error_handler import safe_execute
from src.ui.views.panels.asset_view import AssetBrowserPanelView

class AssetController:
    """
    Coordinates data flow for the Asset Browser panel.
    Handles the asynchronous loading and instantiation of external 3D models and textures.
    """
    def __init__(self) -> None:
        self.view = AssetBrowserPanelView(controller=self)
        ctx.events.subscribe(AppEvent.ASSET_BROWSER_NEEDS_REFRESH, self.refresh_view)
        ctx.events.subscribe(AppEvent.ENTITY_SELECTED, self.on_global_selection)

    def refresh_view(self) -> None:
        models = ctx.engine.get_project_models()
        textures = ctx.engine.get_project_textures()
        self.view.build_asset_lists(models, textures)

    def on_global_selection(self, entity_id: int) -> None:
        data = ctx.engine.get_selected_entity_data()
        path_to_find = data["mesh"]["mat_tex_paths"].get("map_diffuse", "") if data and data.get("mesh") else ""
        self.view.highlight_texture(path_to_find)

    @safe_execute(context="Import Model")
    def import_model(self, path: str) -> None:
        file_name = path.split('/')[-1] if '/' in path else path.split('\\')[-1]
        
        progress = QProgressDialog(f"Loading model into memory: {file_name}", None, 0, 0, ctx.main_window)
        progress.setWindowTitle("Status")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        
        QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)

        try:
            if hasattr(ctx, 'main_window') and hasattr(ctx.main_window, 'gl_widget'):
                ctx.main_window.gl_widget.makeCurrent()
                ctx.engine.preload_model_to_cache(path)
                ctx.main_window.gl_widget.doneCurrent()
            else:
                ctx.engine.preload_model_to_cache(path)

            ctx.engine.import_project_model(path)
            ctx.events.emit(AppEvent.ASSET_BROWSER_NEEDS_REFRESH)
        finally:
            progress.close()

    @safe_execute(context="Import Texture")
    def import_texture(self, path: str) -> None:
        ctx.engine.import_project_texture(path)
        ctx.events.emit(AppEvent.ASSET_BROWSER_NEEDS_REFRESH)

    @safe_execute(context="Delete Asset")
    def request_delete_asset(self, path: str, asset_type: str) -> None:
        if asset_type == 'TEXTURE' and ctx.engine.is_texture_in_use(path):
            raise SimulationError("Cannot delete: Texture is currently applied to a material in the scene!")
            
        ctx.engine.delete_project_asset(path, asset_type)
        ctx.events.emit(AppEvent.ASSET_BROWSER_NEEDS_REFRESH)

    @safe_execute(context="Spawn Model")
    def spawn_model(self, path: str) -> None:
        file_name = path.split('/')[-1] if '/' in path else path.split('\\')[-1]
        
        progress = QProgressDialog(f"Instantiating model: {file_name}", None, 0, 0, ctx.main_window)
        progress.setWindowTitle("Status")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        
        QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)

        try:
            ctx.events.emit(AppEvent.ACTION_BEFORE_MUTATION)
            
            if hasattr(ctx, 'main_window') and hasattr(ctx.main_window, 'gl_widget'):
                ctx.main_window.gl_widget.makeCurrent()
                ctx.engine.spawn_model_from_path(path)
                ctx.main_window.gl_widget.doneCurrent()
            else:
                ctx.engine.spawn_model_from_path(path)
                
            ctx.events.emit(AppEvent.HIERARCHY_NEEDS_REFRESH)
            ctx.events.emit(AppEvent.ENTITY_SELECTED, ctx.engine.get_selected_entity_id())
            ctx.events.emit(AppEvent.SCENE_CHANGED)
            ctx.events.emit(AppEvent.ASSET_BROWSER_NEEDS_REFRESH)
        finally:
            progress.close()

    @safe_execute(context="Apply Texture")
    def apply_texture(self, path: str, map_attr: str) -> None:
        if ctx.engine.get_selected_entity_id() < 0:
            raise SimulationError("Please select an entity in the scene first!")
            
        ctx.events.emit(AppEvent.ACTION_BEFORE_MUTATION)
        if hasattr(ctx, 'main_window') and hasattr(ctx.main_window, 'gl_widget'):
            ctx.main_window.gl_widget.makeCurrent()
            ctx.engine.load_texture_to_selected(map_attr, path)
            ctx.main_window.gl_widget.doneCurrent()
        else:
            ctx.engine.load_texture_to_selected(map_attr, path)
            
        ctx.events.emit(AppEvent.ENTITY_SELECTED, ctx.engine.get_selected_entity_id()) 
        ctx.events.emit(AppEvent.SCENE_CHANGED)