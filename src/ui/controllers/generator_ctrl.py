"""
Generator Controller.

Manages the Synthetic Data Generation UI panel.
Coordinates the live preview loop, asynchronous dataset batch generation, 
and automated CV benchmark orchestration via the Engine Facade.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from PySide6.QtWidgets import QMessageBox, QFileDialog, QApplication
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QOpenGLContext

from src.app import ctx
from src.ui.error_handler import safe_execute
from src.ui.views.panels.generator_view import GeneratorPanelView


class GeneratorController:
    """Coordinates interaction between the Dataset Generator UI and the Engine Backend."""

    def __init__(self) -> None:
        self.view = GeneratorPanelView(controller=self)
        self._last_payload: Optional[Dict[str, Any]] = None
        self._last_generated_dir: Optional[str] = None  
        
        self.preview_timer = QTimer()
        self.preview_timer.setTimerType(Qt.PreciseTimer)
        self.preview_timer.timeout.connect(self._on_playback_tick)
        
        self.is_playing: bool = False
        self.sim_time: float = 0.0
        self.sim_frame: int = 0
        self.sim_dt: float = 0.01
        self.sim_total_frames: int = 0
        
        self._last_real_time: float = 0.0

    def _ensure_preview_mode(self) -> None:
        """Forces the UI to switch to the Data Preview tab context."""
        if hasattr(ctx, "main_window"):
            if hasattr(ctx.main_window, "stacked_view") and ctx.main_window.stacked_view.currentIndex() != 1:
                ctx.main_window.stacked_view.setCurrentIndex(1)
            if hasattr(ctx.main_window, "mode_selector") and ctx.main_window.mode_selector.currentIndex() != 1:
                ctx.main_window.mode_selector.setCurrentIndex(1)

    def _set_main_window_status(self, text: str) -> None:
        """Updates the global status bar label."""
        if hasattr(ctx, 'main_window') and hasattr(ctx.main_window, 'lbl_preview_status'):
            ctx.main_window.lbl_preview_status.setText(text)

    def _set_main_window_time(self, current_time: float) -> None:
        """Updates the global timeline indicator."""
        if hasattr(ctx, 'main_window') and hasattr(ctx.main_window, 'lbl_preview_time'):
            ctx.main_window.lbl_preview_time.setText(f"Time: {current_time:.2f}s")

    def _update_main_window_stats(self, payload: Dict[str, Any]) -> None:
        """Refreshes rendering statistics (e.g., active object count)."""
        if not payload: 
            return
        stats = payload.get("stats", {})
        num_obj = stats.get('num_objects', len(payload.get("objects", [])))
        if hasattr(ctx, 'main_window') and hasattr(ctx.main_window, 'lbl_preview_stats'):
            ctx.main_window.lbl_preview_stats.setText(f"Obj: {num_obj}")

    def _push_to_viewport(self, payload: Dict[str, Any]) -> None:
        """Forwards the rendered FBO pixel payload to the UI viewport canvas."""
        if not payload: 
            return
        self._last_payload = payload
        if hasattr(ctx, "main_window") and hasattr(ctx.main_window, "preview_viewport"):
            vp = ctx.main_window.preview_viewport
            if hasattr(vp, "update_frame"):
                vp.update_frame(payload)
        self._update_main_window_stats(payload)

    def refresh_preview_display(self) -> None:
        """Forces an immediate redraw of the preview viewport."""
        self.handle_preview_once()

    def _run_preview_render(self, w: int, h: int, active_mode: str = "RGB", is_playing: bool = False) -> Optional[Dict[str, Any]]:
        """Handles the OpenGL context lifecycle and commands the Engine Facade to render a preview frame."""
        payload = None
        has_gl = hasattr(ctx, "main_window") and hasattr(ctx.main_window, "gl_widget")
        viewport = getattr(ctx.main_window, "preview_viewport", None) if hasattr(ctx, "main_window") else None
        show_bbox = viewport.is_bbox_enabled() if hasattr(viewport, 'is_bbox_enabled') else True

        context_widget = None
        if has_gl and hasattr(ctx.main_window.gl_widget, "makeCurrent"):
            context_widget = ctx.main_window.gl_widget

        context_acquired = False
        if context_widget is not None:
            try:
                context_widget.makeCurrent()
                context_acquired = QOpenGLContext.currentContext() is not None
            except Exception:
                context_acquired = False

        if not context_acquired:
            return {}
            
        try:
            payload = ctx.engine.get_synthetic_preview(w, h, active_mode, is_playing, show_bbox)
        finally:
            if context_acquired and context_widget is not None and hasattr(context_widget, "doneCurrent"):
                try:
                    context_widget.doneCurrent()
                except Exception:
                    pass
                        
        return payload

    @safe_execute(context="Preview One Frame")
    def handle_preview_once(self) -> None:
        """Renders and displays a single static preview frame."""
        self._ensure_preview_mode()
        
        settings = self.view.get_settings()
        viewport = getattr(ctx.main_window, "preview_viewport", None)
        
        res_w, res_h = viewport.get_resolution() if hasattr(viewport, 'get_resolution') else (settings["res_w"], settings["res_h"])
        active_mode = viewport.get_preview_mode() if hasattr(viewport, 'get_preview_mode') else "RGB"
        
        payload = self._run_preview_render(
            res_w, res_h, 
            active_mode=active_mode,
            is_playing=False 
        )
        self._push_to_viewport(payload)

    def toggle_preview_playback(self) -> bool:
        """Starts or stops the continuous timeline preview simulation."""
        self._ensure_preview_mode()
        
        if self.is_playing:
            self.preview_timer.stop()
            self.is_playing = False
            self.view.set_preview_state(False)
            self._set_main_window_status("Status: Idle")
        else:
            settings = self.view.get_settings()
            self.sim_dt = settings.get("dt", 0.033)
            
            if self.sim_frame >= self.sim_total_frames or self.sim_frame == 0:
                self.sim_total_frames = settings.get("num_frames", 150)
                self.sim_frame = 0
                self.sim_time = 0.0
                
            self._last_real_time = time.time()
            self.preview_timer.start(int(self.sim_dt * 1000)) 
            self.is_playing = True
            self.view.set_preview_state(True)
            self._set_main_window_status("Status: Live Preview")
            
        return self.is_playing

    def stop_preview_playback(self) -> None:
        """Halt the preview playback and resets simulation pointers."""
        self.preview_timer.stop()
        self.is_playing = False
        self.sim_time = 0.0
        self.sim_frame = 0
        self.view.set_preview_state(False)
        self._set_main_window_status("Status: Idle")
        
        animator = getattr(ctx.engine, "animator", None)
        if animator:
            animator.evaluate(0.0, 0.0, target_entity_id=-1)
            
        self.handle_preview_once() 

    def _on_playback_tick(self) -> None:
        """Fired by the QTimer to process the next simulation step."""
        viewport = getattr(ctx.main_window, "preview_viewport", None)
        if viewport and hasattr(viewport, "isVisible") and not viewport.isVisible():
            self.stop_preview_playback()
            return

        current_real_time = time.time()
        elapsed = current_real_time - self._last_real_time
        
        if elapsed < self.sim_dt * 0.9:
            return

        steps_due = max(1, int(elapsed / max(self.sim_dt, 1e-6)))
        steps_due = min(steps_due, 5)
        remaining_steps = self.sim_total_frames - self.sim_frame
        
        if remaining_steps <= 0:
            self.stop_preview_playback()
            return
            
        steps_due = min(steps_due, remaining_steps)
        self._last_real_time = current_real_time

        if self.sim_frame >= self.sim_total_frames:
            self.stop_preview_playback()
            return

        render_time = self.sim_time + self.sim_dt * (steps_due - 1)
        animator = getattr(ctx.engine, "animator", None)
        if animator:
            animator.evaluate(render_time, self.sim_dt * steps_due, target_entity_id=-1)

        settings = self.view.get_settings()
        res_w, res_h = viewport.get_resolution() if hasattr(viewport, 'get_resolution') else (settings["res_w"], settings["res_h"])
        active_mode = viewport.get_preview_mode() if hasattr(viewport, 'get_preview_mode') else "RGB"
       
        try:
            payload = self._run_preview_render(
                res_w, res_h, 
                active_mode=active_mode,
                is_playing=True 
            )
        except Exception:
            self.stop_preview_playback()
            return
            
        self._push_to_viewport(payload)
        self._set_main_window_time(render_time)

        self.sim_time = render_time + self.sim_dt
        self.sim_frame += steps_due

    @safe_execute(context="Browse Directory")
    def handle_browse_directory(self) -> None:
        """Opens a file dialog to select the dataset output directory."""
        path = QFileDialog.getExistingDirectory(ctx.main_window, "Select Dataset Output Directory")
        if path:
            self.view.set_directory(path)

    @safe_execute(context="Auto Detect Duration")
    def handle_auto_duration(self) -> None:
        """Scans the scene via Facade to locate the longest animation track duration."""
        max_duration = ctx.engine.get_max_animation_duration()
        if max_duration > 0:
            self.view.set_duration(max_duration)
        else:
            self.view.set_duration(1.0)

    @safe_execute(context="Dataset Generation")
    def handle_start_generation(self) -> None:
        """Orchestrates the synchronous bulk-generation of the dataset."""
        settings = self.view.get_settings()

        try:
            self.view.set_ui_locked(True)
            total_frames = settings["num_frames"]
            
            self.view.set_progress(0, total_frames, "Initializing...")
            self._set_main_window_status("Status: Rendering Data...")
            
            def progress_callback(frame_idx: int, preview_payload: Optional[Dict[str, Any]] = None, stats: Optional[Dict[str, Any]] = None) -> None:
                self.view.set_progress(frame_idx, total_frames, f"Rendering {frame_idx}/{total_frames} frames")
                if preview_payload:
                    self._ensure_preview_mode()
                    self._push_to_viewport(preview_payload)
                QApplication.processEvents()
                
                if hasattr(ctx, 'main_window') and hasattr(ctx.main_window, 'gl_widget'):
                    ctx.main_window.gl_widget.makeCurrent()

            if hasattr(ctx, 'main_window') and hasattr(ctx.main_window, 'gl_widget'):
                ctx.main_window.gl_widget.makeCurrent()

            final_path = ctx.engine.run_synthetic_generation(settings, progress_callback)
            self._last_generated_dir = final_path 
            
            self.view.set_progress(total_frames, total_frames, "Completed!")
            self._set_main_window_status("Status: Idle")
            
            QMessageBox.information(ctx.main_window, "Success", f"Dataset synthesized successfully.\nSaved to:\n{final_path}")
            self.view.set_status("Generation Complete.")
            self.view.set_progress(0, 0, "") 
            
        finally:
            if hasattr(ctx, 'main_window') and hasattr(ctx.main_window, 'gl_widget'):
                try:
                    ctx.main_window.gl_widget.doneCurrent()
                except Exception:
                    pass
            self.view.set_ui_locked(False)

    @safe_execute(context="CV Benchmark")
    def handle_run_cv_benchmark(self) -> None:
        """Validates the exported dataset and launches the YOLO training pipeline via Facade."""
        self._ensure_preview_mode()

        if self.is_playing:
            self.stop_preview_playback()

        settings = self.view.get_settings()
        selected_output = str(settings.get("output_dir", "")).strip()
        
        if selected_output:
            dataset_dir = Path(selected_output).resolve()
        else:
            if not self._last_generated_dir:
                raise FileNotFoundError(
                    "Missing dataset directory. Please specify an output directory or click 'START BATCH GENERATION' first."
                )
            dataset_dir = Path(self._last_generated_dir).resolve()

        dataset_yaml = dataset_dir / "dataset.yaml"
        if not dataset_yaml.exists():
            raise FileNotFoundError(
                f"Missing dataset descriptor at:\n{dataset_yaml}\n"
                "Please generate a dataset first in Preview mode."
            )

        task = str(settings.get("cv_task", "auto")).strip().lower()
        model_name = str(settings.get("cv_model", "")).strip() or None
        run_training = not bool(settings.get("cv_no_train", True))

        benchmark_output = dataset_dir.parent / "cv_benchmark_ui" / datetime.now().strftime("%Y%m%d_%H%M%S")

        config_dict = {
            "model_type": model_name,
            "task": task if task in {"auto", "detect", "segment"} else "auto",
            "epochs": max(1, int(settings.get("cv_epochs", 3))),
            "batch_size": max(1, int(settings.get("cv_batch", 8))),
            "imgsz": max(32, int(settings.get("cv_imgsz", 640))),
            "confidence_threshold": float(settings.get("cv_conf", 0.25)),
            "run_training": run_training,
            "split_ratios": settings.get("cv_split_ratios", (0.7, 0.2, 0.1))
        }

        try:
            self.view.set_ui_locked(True)
            self.view.set_progress(0, 1, "Running CV benchmark...")
            self._set_main_window_status("Status: Benchmark Running...")
            QApplication.processEvents()

            result = ctx.engine.run_cv_benchmark(config_dict, dataset_dir, benchmark_output)
            
            records = result.get("records", []) or []
            record = records[0] if records else {}
            status = str(record.get("status", "unknown"))
            metric_name = str(record.get("primary_metric_name", "box_map50"))
            metric_value = float(record.get("primary_metric", 0.0))
            vis_frames = int(record.get("visualized_frames", 0))
            total_frames = int(record.get("dataset_total_frames", 0))
            match_p = float(record.get("match_precision", 0.0))
            match_r = float(record.get("match_recall", 0.0))
            artifacts = result.get("artifacts", {}) or {}

            self.view.set_progress(1, 1, "CV benchmark completed.")
            self.view.set_status("CV benchmark completed.")
            self._set_main_window_status("Status: Idle")

            if status != "ok":
                err = record.get("error", "Unknown error")
                QMessageBox.warning(
                    ctx.main_window,
                    "CV Benchmark Failed",
                    f"Benchmark failed for dataset:\n{dataset_dir}\n\nError:\n{err}",
                )
                return

            QMessageBox.information(
                ctx.main_window,
                "CV Benchmark Completed",
                (
                    f"Dataset: {dataset_dir}\n"
                    f"Task: {record.get('task', task)}\n"
                    f"{metric_name}: {metric_value:.4f}\n\n"
                    f"Frame comparison (GT vs Pred): {vis_frames}/{total_frames}\n"
                    f"Match Precision: {match_p:.4f}\n"
                    f"Match Recall: {match_r:.4f}\n\n"
                    f"Summary: {artifacts.get('summary_md', '')}\n"
                    f"CSV: {artifacts.get('csv', '')}\n"
                    f"JSON: {artifacts.get('json', '')}\n"
                    f"Comparisons: {record.get('comparison_dir', '')}\n"
                    f"Frame Compare CSV: {record.get('comparison_csv', '')}"
                ),
            )
        finally:
            self.view.set_progress(0, 0, "")
            self.view.set_ui_locked(False)
            self._set_main_window_status("Status: Idle")

    def handle_run_cv_proof(self) -> None:
        """Legacy method kept for compatibility with older UI signal wiring."""
        self.handle_run_cv_benchmark()