import os
from typing import Optional
from PySide6.QtWidgets import QFileDialog, QMessageBox, QInputDialog, QDialog, QProgressDialog, QApplication, QProgressBar
from PySide6.QtCore import Qt, QEventLoop

from src.app import ctx, AppEvent, config
from src.ui.error_handler import safe_execute
from src.ui.views.project_browser import ProjectBrowserDialog

class ProjectController:
    """
    Manages the application lifecycle workflow: Undo/Redo, Save/Load JSON, and Export OBJ.
    """
    def __init__(self) -> None:
        self.projects_dir = config.PROJECTS_DIR
        os.makedirs(self.projects_dir, exist_ok=True)
            
        self.current_project_name: Optional[str] = None
        self.undo_stack: list[str] = []
        self.redo_stack: list[str] = []
        self.max_history = config.MAX_UNDO_HISTORY
        self.is_restoring = False 
        
        ctx.events.subscribe(AppEvent.ACTION_BEFORE_MUTATION, self.record_history)

    def _update_window_title(self) -> None:
        if not hasattr(ctx, 'main_window'): return
        base_title = config.APP_TITLE
        if self.current_project_name:
            ctx.main_window.setWindowTitle(f"{base_title} - [{self.current_project_name}]")
        else:
            ctx.main_window.setWindowTitle(f"{base_title} - [Unsaved]")

    def record_history(self) -> None:
        if self.is_restoring: return
        snapshot = ctx.engine.get_scene_snapshot()
        if self.undo_stack and self.undo_stack[-1] == snapshot: return
        
        self.undo_stack.append(snapshot)
        if len(self.undo_stack) > self.max_history: 
            self.undo_stack.pop(0) 
            
        self.redo_stack.clear() 
        ctx.events.emit(AppEvent.HISTORY_RECORDED)

    @safe_execute(context="Undo / Redo Restoration")
    def restore_snapshot(self, snapshot_str: str) -> None:
        if not snapshot_str: return
        self.is_restoring = True
        
        if hasattr(ctx, 'main_window') and hasattr(ctx.main_window, 'gl_widget'):
            ctx.main_window.gl_widget.makeCurrent()
            current_aspect = ctx.main_window.gl_widget.width() / max(ctx.main_window.gl_widget.height(), 1)
            ctx.engine.restore_snapshot(snapshot_str, current_aspect)
            ctx.main_window.gl_widget.doneCurrent()
        else:
            ctx.engine.restore_snapshot(snapshot_str, config.DEFAULT_ASPECT_RATIO)
        
        ctx.events.emit(AppEvent.HIERARCHY_NEEDS_REFRESH)
        current_id = ctx.engine.get_selected_entity_id()
        ctx.events.emit(AppEvent.ENTITY_SELECTED, current_id)
        ctx.events.emit(AppEvent.SCENE_CHANGED)
            
        self.is_restoring = False

    def undo(self) -> None:
        if not self.undo_stack: return
        self.redo_stack.append(ctx.engine.get_scene_snapshot())
        self.restore_snapshot(self.undo_stack.pop())

    def redo(self) -> None:
        if not self.redo_stack: return
        self.undo_stack.append(ctx.engine.get_scene_snapshot())
        self.restore_snapshot(self.redo_stack.pop())

    @safe_execute(context="New Project")
    def new_project(self) -> None:
        ans = QMessageBox.question(ctx.main_window, "New Project", 
                                   "Creating a new project will clear the current scene. Continue?", 
                                   QMessageBox.Yes | QMessageBox.No)
        if ans == QMessageBox.No: return
        
        if hasattr(ctx, 'main_window') and hasattr(ctx.main_window, 'gl_widget'):
            ctx.main_window.gl_widget.makeCurrent()
            ctx.engine.clear_scene()
            ctx.engine.auto_load_default_assets(config.TEXTURES_DIR)
            ctx.main_window.gl_widget.bg_color = config.DEFAULT_BG_COLOR 
            ctx.main_window.gl_widget.doneCurrent()
        else:
            ctx.engine.clear_scene()
            ctx.engine.auto_load_default_assets(config.TEXTURES_DIR)
        
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.current_project_name = None
        self._update_window_title()
        
        ctx.events.emit(AppEvent.PROJECT_LOADED)
        ctx.events.emit(AppEvent.HIERARCHY_NEEDS_REFRESH)
        ctx.events.emit(AppEvent.ASSET_BROWSER_NEEDS_REFRESH)
        current_id = ctx.engine.get_selected_entity_id()
        ctx.events.emit(AppEvent.ENTITY_SELECTED, current_id)
        ctx.events.emit(AppEvent.SCENE_CHANGED)

    @safe_execute(context="Save Project")
    def save_project(self) -> None:
        name = self.current_project_name
        if not name:
            name, ok = QInputDialog.getText(ctx.main_window, "Save Project", "Enter Project Name:")
            if not ok or not name.strip(): return
            name = "".join([c for c in name.strip() if c.isalnum() or c in (' ', '-', '_')])
            
            file_path = os.path.join(self.projects_dir, f"{name}.json")
            if os.path.exists(file_path):
                ans = QMessageBox.question(ctx.main_window, "Confirm Overwrite", 
                                           "This project already exists. Overwrite?", 
                                           QMessageBox.Yes | QMessageBox.No)
                if ans == QMessageBox.No: return
        else:
            file_path = os.path.join(self.projects_dir, f"{name}.json")
        
        metadata = {
            "render_mode": ctx.main_window.cmb_render.currentIndex() if hasattr(ctx.main_window, 'cmb_render') else 1,
            "wireframe": ctx.main_window.chk_wire.isChecked() if hasattr(ctx.main_window, 'chk_wire') else False
        }
        if hasattr(ctx, 'main_window') and hasattr(ctx.main_window, 'gl_widget'):
            bg = ctx.main_window.gl_widget.bg_color
            metadata["bg_color"] = [bg[0], bg[1], bg[2]]
        
        ctx.engine.save_project(file_path, metadata)
        
        self.current_project_name = name
        self._update_window_title()
        ctx.events.emit(AppEvent.PROJECT_SAVED)
        QMessageBox.information(ctx.main_window, "Success", f"Project '{name}' saved successfully!")

    @safe_execute(context="Load Project")
    def load_project(self) -> None:
        dialog = ProjectBrowserDialog(ctx.main_window, self.projects_dir)
        if dialog.exec() == QDialog.Accepted and dialog.selected_path:
            selected_path = dialog.selected_path
            
            QApplication.processEvents()
            
            progress = QProgressDialog("Loading Project and Assets...", None, 0, 0, ctx.main_window)
            progress.setWindowTitle("Status")
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            
            # Ẩn thanh progress bar, chỉ để lại Text
            bar = progress.findChild(QProgressBar)
            if bar:
                bar.hide()
                
            progress.show()
            progress.repaint()
            QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)

            try:
                if hasattr(ctx, 'main_window') and hasattr(ctx.main_window, 'gl_widget'):
                    ctx.main_window.gl_widget.makeCurrent()
                    current_aspect = ctx.main_window.gl_widget.width() / max(ctx.main_window.gl_widget.height(), 1)
                    # Gọi lại hàm load_project cũ (không cần truyền progress_callback nữa)
                    metadata = ctx.engine.load_project(selected_path, current_aspect)
                    ctx.main_window.gl_widget.doneCurrent()
                else:
                    metadata = ctx.engine.load_project(selected_path, config.DEFAULT_ASPECT_RATIO)
                
                if metadata and hasattr(ctx, 'main_window'):
                    if "bg_color" in metadata and hasattr(ctx.main_window.gl_widget, 'bg_color'):
                        bg = metadata["bg_color"]
                        ctx.main_window.gl_widget.bg_color = (bg[0], bg[1], bg[2]) 
                        
                    if "render_mode" in metadata and hasattr(ctx.main_window, 'cmb_render'):
                        idx = metadata["render_mode"]
                        ctx.main_window.cmb_render.setCurrentIndex(idx if idx <= 1 else 1)
                        
                    if "wireframe" in metadata and hasattr(ctx.main_window, 'chk_wire'):
                        ctx.main_window.chk_wire.setChecked(metadata["wireframe"])

                self.undo_stack.clear()
                self.redo_stack.clear()
                self.current_project_name = os.path.splitext(os.path.basename(selected_path))[0]
                self._update_window_title()
                
                ctx.events.emit(AppEvent.PROJECT_LOADED)
                ctx.events.emit(AppEvent.HIERARCHY_NEEDS_REFRESH)
                ctx.events.emit(AppEvent.ASSET_BROWSER_NEEDS_REFRESH)
                current_id = ctx.engine.get_selected_entity_id()
                ctx.events.emit(AppEvent.ENTITY_SELECTED, current_id)
                ctx.events.emit(AppEvent.SCENE_CHANGED)
            finally:
                progress.close()

    @safe_execute(context="Export OBJ")
    def export_obj(self) -> None:
        parent_dir = QFileDialog.getExistingDirectory(ctx.main_window, "Select Export Directory")
        if not parent_dir: return
        
        folder_name, ok = QInputDialog.getText(ctx.main_window, "Export Folder Name", "Folder name:", text=config.DEFAULT_EXPORT_FOLDER)
        if not ok or not folder_name.strip(): return
        
        safe_folder_name = "".join([c for c in folder_name.strip() if c.isalnum() or c in (' ', '-', '_')])
        export_dir = os.path.join(parent_dir, safe_folder_name)
        os.makedirs(export_dir, exist_ok=True)

        ctx.engine.export_scene_obj(export_dir)
        QMessageBox.information(ctx.main_window, "Success", f"Scene exported successfully to:\n{export_dir}\n(Includes .obj and .mtl)")