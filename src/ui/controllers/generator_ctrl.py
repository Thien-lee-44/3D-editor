import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from PySide6.QtWidgets import QMessageBox, QFileDialog, QApplication
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QOpenGLContext
import numpy as np  

from src.app import ctx
from src.ui.error_handler import safe_execute
from src.ui.views.panels.generator_view import GeneratorPanelView
from src.engine.synthetic.generator import SyntheticDataGenerator
from src.engine.synthetic.cv_benchmark import CVBenchmarkRunner, CVBenchmarkConfig

class GeneratorController:
    def __init__(self) -> None:
        self.view = GeneratorPanelView(controller=self)
        self.generator_backend: Optional[SyntheticDataGenerator] = None
        self._last_payload: Optional[Dict[str, Any]] = None
        
        self.preview_timer = QTimer()
        self.preview_timer.setTimerType(Qt.PreciseTimer)
        self.preview_timer.timeout.connect(self._on_playback_tick)
        
        self.is_playing: bool = False
        self.sim_time: float = 0.0
        self.sim_frame: int = 0
        self.sim_dt: float = 0.01
        self.sim_total_frames: int = 0
        
        self._last_real_time: float = 0.0

    def _ensure_backend(self) -> None:
        if self.generator_backend is None:
            self.generator_backend = SyntheticDataGenerator(ctx.engine)

    def _ensure_preview_mode(self) -> None:
        if hasattr(ctx, "main_window"):
            if hasattr(ctx.main_window, "stacked_view") and ctx.main_window.stacked_view.currentIndex() != 1:
                ctx.main_window.stacked_view.setCurrentIndex(1)
            if hasattr(ctx.main_window, "mode_selector") and ctx.main_window.mode_selector.currentIndex() != 1:
                ctx.main_window.mode_selector.setCurrentIndex(1)

    def _set_main_window_status(self, text: str) -> None:
        if hasattr(ctx, 'main_window') and hasattr(ctx.main_window, 'lbl_preview_status'):
            ctx.main_window.lbl_preview_status.setText(text)

    def _set_main_window_time(self, current_time: float) -> None:
        if hasattr(ctx, 'main_window') and hasattr(ctx.main_window, 'lbl_preview_time'):
            ctx.main_window.lbl_preview_time.setText(f"Time: {current_time:.2f}s")

    def _update_main_window_stats(self, payload: Dict[str, Any]) -> None:
        if not payload: return
        stats = payload.get("stats", {})
        num_obj = stats.get('num_objects', len(payload.get("objects", [])))
        if hasattr(ctx, 'main_window') and hasattr(ctx.main_window, 'lbl_preview_stats'):
            ctx.main_window.lbl_preview_stats.setText(f"Obj: {num_obj}")

    def _push_to_viewport(self, payload: Dict[str, Any]) -> None:
        if not payload: 
            return
        self._last_payload = payload
        if hasattr(ctx, "main_window") and hasattr(ctx.main_window, "preview_viewport"):
            vp = ctx.main_window.preview_viewport
            if hasattr(vp, "update_frame"):
                vp.update_frame(payload)
        self._update_main_window_stats(payload)

    def refresh_preview_display(self) -> None:
        self.handle_preview_once()

    def _run_preview_render(self, w: int, h: int, active_mode: str = "RGB", is_playing: bool = False) -> Optional[Dict[str, Any]]:
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
            payload = self.generator_backend.extract_preview_frame(
                w,
                h,
                active_mode,
                is_playing,
                show_bbox=show_bbox,
            )
        finally:
            if context_acquired and context_widget is not None and hasattr(context_widget, "doneCurrent"):
                try:
                    context_widget.doneCurrent()
                except Exception:
                    pass
                        
        return payload

    @safe_execute(context="Preview One Frame")
    def handle_preview_once(self) -> None:
        self._ensure_backend()
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
        self._ensure_backend()
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
        else:
            self.view.set_duration(1.0)

    @safe_execute(context="Dataset Generation")
    def handle_start_generation(self) -> None:
        settings = self.view.get_settings()
        self._ensure_backend()

        try:
            target_dir = settings["output_dir"] if settings["output_dir"] else None
            self.generator_backend.setup_directories(target_dir)
            self.view.set_ui_locked(True)
            total_frames = settings["num_frames"]
            
            self.view.set_progress(0, total_frames, "Initializing...")
            self._set_main_window_status("Status: Rendering Data...")
            preview_stride = max(1, total_frames // 120)
            
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
                self.generator_backend.generate_batch(
                    num_frames=total_frames, 
                    dt=settings["dt"],
                    res_w=settings["res_w"],
                    res_h=settings["res_h"],
                    use_rand_light=settings["use_rand_light"],
                    use_rand_cam=settings["use_rand_cam"],
                    progress_cb=progress_callback,
                    preview_stride=preview_stride, 
                )
            else:
                self.generator_backend.generate_batch(
                    num_frames=total_frames, 
                    dt=settings["dt"],
                    res_w=settings["res_w"],
                    res_h=settings["res_h"],
                    use_rand_light=settings["use_rand_light"],
                    use_rand_cam=settings["use_rand_cam"],
                    progress_cb=progress_callback,
                    preview_stride=preview_stride, 
                )
            
            final_path = self.generator_backend.output_dir
            self.view.set_progress(total_frames, total_frames, "Completed!")
            self._set_main_window_status("Status: Idle")
            
            QMessageBox.information(ctx.main_window, "Success", f"Dataset synthesized successfully.\nSaved to:\n{final_path}")
            self.view.set_status("Generation Complete.")
            self.view.set_progress(0, 0, "") 
            
        finally:
            self.view.set_ui_locked(False)

    @safe_execute(context="CV Benchmark")
    def handle_run_cv_benchmark(self) -> None:
        self._ensure_backend()
        self._ensure_preview_mode()

        if self.is_playing:
            self.stop_preview_playback()

        settings = self.view.get_settings()
        selected_output = str(settings.get("output_dir", "")).strip()
        if selected_output:
            dataset_dir = Path(selected_output).resolve()
        else:
            dataset_dir = self.generator_backend.output_dir.resolve()

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
        variant_name = dataset_dir.name or "dataset"

        config = CVBenchmarkConfig(
            model_type=model_name,
            task=task if task in {"auto", "detect", "segment"} else "auto",
            epochs=max(1, int(settings.get("cv_epochs", 3))),
            batch_size=max(1, int(settings.get("cv_batch", 8))),
            imgsz=max(32, int(settings.get("cv_imgsz", 640))),
            confidence_threshold=float(settings.get("cv_conf", 0.25)),
            run_training=run_training,
        )
        runner = CVBenchmarkRunner(output_dir=benchmark_output, config=config)

        try:
            self.view.set_ui_locked(True)
            self.view.set_progress(0, 1, "Running CV benchmark...")
            self._set_main_window_status("Status: Benchmark Running...")
            QApplication.processEvents()

            result = runner.run({variant_name: dataset_dir})
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

    # Legacy method kept for compatibility with older UI wiring.
    def handle_run_cv_proof(self) -> None:
        self.handle_run_cv_benchmark()
