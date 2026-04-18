from PySide6.QtWidgets import QMessageBox, QFileDialog, QProgressDialog
from PySide6.QtCore import Qt
from src.app import ctx
from src.ui.error_handler import safe_execute
from src.ui.views.panels.generator_view import GeneratorDialogView
from src.engine.synthetic.generator import SyntheticDataGenerator

class GeneratorController:
    """
    Coordinates data flow and business logic for the Synthetic Data Generator dialog.
    Acts as the bridge between the PySide6 UI and the headless dataset generation backend.
    """
    def __init__(self) -> None:
        self.view = GeneratorDialogView(controller=self)
        self.generator_backend = None

    def show_dialog(self) -> None:
        """Presents the dialog to the user and brings it to the foreground."""
        self.view.show()
        self.view.raise_()
        self.view.activateWindow()

    @safe_execute(context="Browse Directory")
    def handle_browse_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(ctx.main_window, "Select Dataset Output Directory")
        if path:
            self.view.set_directory(path)

    @safe_execute(context="Auto Detect Duration")
    def handle_auto_duration(self) -> None:
        from src.engine.scene.components.animation_cmp import AnimationComponent
        
        max_duration = 0.0
        for ent in ctx.engine.scene.entities:
            anim = ent.get_component(AnimationComponent)
            if anim and anim.duration > max_duration:
                max_duration = anim.duration
                
        if max_duration > 0:
            self.view.set_duration(max_duration)
            QMessageBox.information(ctx.main_window, "Auto-Detect", f"Found maximum animation duration: {max_duration}s")
        else:
            QMessageBox.information(ctx.main_window, "Auto-Detect", "No keyframes found in the scene. Defaulting to 1.0s.")
            self.view.set_duration(1.0)

    @safe_execute(context="Dataset Generation")
    def handle_start_generation(self) -> None:
        settings = self.view.get_settings()

        if self.generator_backend is None:
            self.generator_backend = SyntheticDataGenerator(ctx.engine)

        try:
            # Delegate path resolution to the backend generator
            target_dir = settings["output_dir"] if settings["output_dir"] else None
            self.generator_backend.setup_directories(target_dir)
            
            self.view.set_ui_locked(True)
            
            total_frames = settings["num_frames"]
            progress = QProgressDialog("Synthesizing Dataset...", "Cancel", 0, total_frames, ctx.main_window)
            progress.setWindowTitle("Data Generator")
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0) 
            progress.setValue(0)
            
            def progress_callback(frame_idx: int) -> None:
                progress.setValue(frame_idx)
                if progress.wasCanceled():
                    self.generator_backend.is_running = False

            # Inject the generation payload into the OpenGL context if available
            if hasattr(ctx, 'main_window') and hasattr(ctx.main_window, 'gl_widget'):
                ctx.main_window.gl_widget.makeCurrent()
                self.generator_backend.generate_batch(
                    num_frames=total_frames, 
                    dt=settings["dt"],
                    res_w=settings["res_w"],
                    res_h=settings["res_h"],
                    use_randomization=settings["use_randomization"],
                    progress_cb=progress_callback
                )
                ctx.main_window.gl_widget.doneCurrent()
            else:
                self.generator_backend.generate_batch(
                    num_frames=total_frames, 
                    dt=settings["dt"],
                    res_w=settings["res_w"],
                    res_h=settings["res_h"],
                    use_randomization=settings["use_randomization"],
                    progress_cb=progress_callback
                )
            
            if progress.wasCanceled():
                QMessageBox.warning(ctx.main_window, "Aborted", "Generation was cancelled by the user.")
            else:
                final_path = self.generator_backend.output_dir
                msg = f"Successfully generated {total_frames} frames!\n\nSaved to:\n{final_path}"
                QMessageBox.information(ctx.main_window, "Success", msg)
            
        except Exception as e:
            QMessageBox.critical(ctx.main_window, "Generation Error", str(e))
        finally:
            self.view.set_ui_locked(False)