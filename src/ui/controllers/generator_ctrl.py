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