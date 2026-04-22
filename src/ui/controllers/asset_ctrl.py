import os
from PySide6.QtWidgets import QProgressDialog, QApplication
from PySide6.QtCore import Qt, QEventLoop

from src.app import ctx, AppEvent
from src.app.exceptions import SimulationError
from src.app.config import TEXTURE_CHANNELS
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

    def _get_timeline(self):
        return getattr(ctx.main_window._controller, 'timeline_ctrl', None) if hasattr(ctx, 'main_window') else None

    def _is_same_texture_assignment(self, map_attr: str, path: str) -> bool:
        data = ctx.engine.get_selected_entity_data()
        if not data or not data.get("mesh"):
            return False
        tex_paths = data["mesh"].get("mat_tex_paths", {})
        current = tex_paths.get(map_attr, "")
        if not isinstance(current, str):
            return False
        return os.path.normcase(os.path.normpath(current)) == os.path.normcase(os.path.normpath(path))

    def _sync_texture_change_to_keyframe(self) -> None:
        timeline = self._get_timeline()
        curr_time = timeline.current_time if timeline else 0.0
        data = ctx.engine.get_selected_entity_data()
        if not data or "mesh" not in data or "mat_tex_paths" not in data["mesh"]:
            return

        is_kf, is_new, t_time = ctx.engine.update_keyframe_property(
            curr_time, "Mesh", "mat_tex_paths", data["mesh"]["mat_tex_paths"]
        )
        if timeline and is_kf:
            if is_new:
                timeline._refresh_dope_sheet()
            if abs(timeline.current_time - t_time) > 0.001:
                timeline.set_time(t_time)
            elif hasattr(ctx.engine, 'animator'):
                ctx.engine.animator.evaluate(t_time, 0.0)

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

        if map_attr not in TEXTURE_CHANNELS.values():
            raise SimulationError(f"Unsupported texture channel: '{map_attr}'")

        selected = ctx.engine.get_selected_entity_data()
        if not selected or not selected.get("mesh"):
            raise SimulationError("Selected entity has no mesh renderer to receive textures.")

        normalized_path = os.path.abspath(path)
        if self._is_same_texture_assignment(map_attr, normalized_path):
            ctx.events.emit(AppEvent.ENTITY_SELECTED, ctx.engine.get_selected_entity_id())
            return

        timeline = self._get_timeline()
        curr_time = timeline.current_time if timeline else 0.0
        if curr_time > 0.01 and hasattr(ctx.engine, 'animator'):
            # Prime animation state before mutating material to preserve base snapshot.
            ctx.engine.animator.evaluate(curr_time, 0.0)

        ctx.events.emit(AppEvent.ACTION_BEFORE_MUTATION)
        if hasattr(ctx, 'main_window') and hasattr(ctx.main_window, 'gl_widget'):
            ctx.main_window.gl_widget.makeCurrent()
            ctx.engine.load_texture_to_selected(map_attr, normalized_path)
            ctx.main_window.gl_widget.doneCurrent()
        else:
            ctx.engine.load_texture_to_selected(map_attr, normalized_path)

        self._sync_texture_change_to_keyframe()
        ctx.events.emit(AppEvent.ENTITY_SELECTED, ctx.engine.get_selected_entity_id()) 
        ctx.events.emit(AppEvent.SCENE_CHANGED)
