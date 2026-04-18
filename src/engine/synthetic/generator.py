import random
import glm
import numpy as np
from pathlib import Path
from typing import Any, Callable, Optional, Dict, List, Tuple

from src.engine.synthetic.exporters.image_writer import ImageWriter
from src.engine.synthetic.exporters.yolo_writer import YOLOWriter
from src.engine.synthetic.label_utils import LabelUtils

from src.engine.scene.components import TransformComponent, MeshRenderer, CameraComponent, LightComponent
from src.engine.scene.components.semantic_cmp import SemanticComponent

class SyntheticDataGenerator:
    """
    The central orchestrator for the Synthetic Dataset Generation pipeline.
    
    Features:
    - Automated temporal advancement and multi-pass frame capture.
    - YOLO Ground Truth extraction with multi-mesh instance merging.
    - Domain Randomization (Lighting & Camera Jitter) to prevent AI overfitting.
    - Flexible output directory management using modern pathlib.
    """

    def __init__(self, engine: Any) -> None:
        self.engine = engine 
        self.scene = engine.scene
        self.renderer = engine.renderer
        self.animator = getattr(engine, 'animator', None)
        
        self.is_running: bool = False
        self.output_dir: Path = Path.cwd() / "datasets" / "synthetic_output" # Default Fallback

    # =========================================================================
    # DIRECTORY & WORKSPACE MANAGEMENT
    # =========================================================================

    def setup_directories(self, custom_path: Optional[str] = None) -> None:
        """
        Prepares the target dataset folder hierarchy and writes the YOLO dataset.yaml.
        
        Args:
            custom_path (Optional[str]): A user-defined export path. If None, uses the default path.
        """
        if custom_path and str(custom_path).strip():
            self.output_dir = Path(custom_path)
        else:
            # Generate a unique default folder if none is provided
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path.cwd() / "datasets" / f"export_{timestamp}"

        # Create subdirectories using pathlib (parents=True acts like mkdir -p)
        (self.output_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "masks").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "depth").mkdir(parents=True, exist_ok=True)

        self._write_dataset_yaml()

    def _write_dataset_yaml(self) -> None:
        """Generates the dataset.yaml required by Ultralytics YOLO."""
        yaml_path = self.output_dir / "dataset.yaml"
        
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write("path: ./  # Adjust to absolute path if moving to Colab/Server\n")
            f.write("train: images\n")
            f.write("val: images\n\n")
            f.write("names:\n")
            
            classes = getattr(self.engine, 'semantic_classes', {0: {"name": "Background"}})
            for c_id, c_info in classes.items():
                name = c_info.get("name", "Unknown") if isinstance(c_info, dict) else c_info
                f.write(f"  {c_id}: {name}\n")

    # =========================================================================
    # CORE GENERATION PIPELINE
    # =========================================================================

    def generate_batch(self, num_frames: int, dt: float, res_w: int = 1024, res_h: int = 1024, 
                       use_randomization: bool = True, progress_cb: Optional[Callable[[int], None]] = None) -> None:
        """
        Executes the synchronous batch generation loop with Domain Randomization.
        """
        if not self.scene or not self.renderer:
            raise RuntimeError("Cannot generate dataset: Core subsystems are missing.")

        active_camera, active_cam_tf = self._get_active_camera()
        if not active_camera or not active_cam_tf:
            raise RuntimeError("Cannot generate dataset: No active camera found in the scene.")

        self.is_running = True
        old_aspect = self._force_camera_aspect(active_camera, res_w, res_h)
        original_light_states = self._backup_lighting_state()

        try:
            current_sim_time = 0.0 

            for frame_idx in range(num_frames):
                if not self.is_running:
                    break

                self._process_ui_events()

                if self.animator:
                    self.animator.evaluate(current_sim_time, dt) 

                pre_jitter_pos = glm.vec3(active_cam_tf.position)

                if use_randomization:
                    self._apply_domain_randomization(original_light_states, active_cam_tf)

                # Fetch matrices after randomization
                view_mat = active_camera.get_view_matrix()
                proj_mat = active_camera.get_projection_matrix()

                if use_randomization:
                    # Instantly reset camera position to prevent exponential jitter drift
                    active_cam_tf.position = pre_jitter_pos

                # Extract AI Ground Truth
                bboxes = self._calculate_bounding_boxes(view_mat, proj_mat, res_w, res_h)

                # Export all data passes
                self._export_frame(frame_idx, bboxes, res_w, res_h)
                
                current_sim_time += dt 

                if progress_cb:
                    progress_cb(frame_idx + 1)

        finally:
            self.is_running = False
            self._restore_lighting_state(original_light_states)
            self._restore_camera_aspect(active_camera, old_aspect)

    # =========================================================================
    # PRIVATE HELPER METHODS (Clean Code Architecture)
    # =========================================================================

    def _get_active_camera(self) -> Tuple[Optional[CameraComponent], Optional[TransformComponent]]:
        for tf, cam, ent in self.scene.cached_cameras:
            if getattr(cam, 'is_active', False):
                return cam, tf
        return None, None

    def _force_camera_aspect(self, cam: CameraComponent, w: int, h: int) -> float:
        old_aspect = getattr(cam, 'aspect_ratio', getattr(cam, 'aspect', 1.0))
        target_aspect = w / h
        if hasattr(cam, 'aspect_ratio'):
            cam.aspect_ratio = target_aspect
        else:
            cam.aspect = target_aspect
            
        if hasattr(cam, 'update_projection_matrix'):
            cam.update_projection_matrix()
        return old_aspect

    def _restore_camera_aspect(self, cam: CameraComponent, old_aspect: float) -> None:
        if hasattr(cam, 'aspect_ratio'):
            cam.aspect_ratio = old_aspect
        else:
            cam.aspect = old_aspect
            
        if hasattr(cam, 'update_projection_matrix'):
            cam.update_projection_matrix()

    def _backup_lighting_state(self) -> List[Dict[str, Any]]:
        states = []
        for l_tf, light, ent in self.scene.cached_lights:
            states.append({
                'comp': light,
                'tf': l_tf,
                'diffuse': glm.vec3(light.diffuse),
                'ambient': glm.vec3(light.ambient),
                'quat_rot': glm.quat(l_tf.quat_rot),
                'rotation': glm.vec3(l_tf.rotation)
            })
        return states

    def _restore_lighting_state(self, states: List[Dict[str, Any]]) -> None:
        for state in states:
            l_comp = state['comp']
            l_tf = state['tf']
            l_comp.diffuse = state['diffuse']
            l_comp.ambient = state['ambient']
            l_tf.quat_rot = state['quat_rot']
            l_tf.rotation = state['rotation']
            if hasattr(l_tf, 'sync_from_gui'):
                l_tf.sync_from_gui()

    def _apply_domain_randomization(self, original_states: List[Dict[str, Any]], cam_tf: TransformComponent) -> None:
        """Injects calculated noise into lighting and camera transforms to prevent AI Overfitting."""
        for state in original_states:
            l_comp = state['comp']
            l_tf = state['tf']
            
            if l_comp.type == "Directional":
                yaw = random.uniform(0.0, 360.0)
                pitch = random.uniform(-80.0, -10.0)
                l_tf.quat_rot = glm.angleAxis(glm.radians(yaw), glm.vec3(0, 1, 0)) * \
                                glm.angleAxis(glm.radians(pitch), glm.vec3(1, 0, 0))
                
                intensity = random.uniform(0.5, 1.5)
                tint_r = random.uniform(0.9, 1.1)
                tint_b = random.uniform(0.9, 1.1)
                l_comp.diffuse = glm.vec3(intensity * tint_r, intensity, intensity * tint_b)
                l_comp.ambient = l_comp.diffuse * 0.2

        jitter = glm.vec3(random.uniform(-0.015, 0.015), random.uniform(-0.015, 0.015), random.uniform(-0.015, 0.015))
        cam_tf.position += jitter

    def _calculate_bounding_boxes(self, view_mat: glm.mat4, proj_mat: glm.mat4, res_w: int, res_h: int) -> List[List[float]]:
        merged_bboxes = {}
        for ent in self.scene.entities:
            semantic = ent.get_component(SemanticComponent)
            tf = ent.get_component(TransformComponent)
            mesh = ent.get_component(MeshRenderer)
            
            if semantic and tf and mesh and mesh.visible and not getattr(mesh, 'is_proxy', False):
                bbox = LabelUtils.get_2d_bounding_box(tf, mesh.geometry, view_mat, proj_mat, res_w, res_h)
                if bbox:
                    xmin, ymin, xmax, ymax = bbox
                    merge_key = f"{semantic.track_id}_{semantic.class_id}"
                    
                    if merge_key in merged_bboxes:
                        curr = merged_bboxes[merge_key]
                        merged_bboxes[merge_key] = [
                            semantic.class_id, 
                            min(curr[1], xmin), min(curr[2], ymin),
                            max(curr[3], xmax), max(curr[4], ymax)
                        ]
                    else:
                        merged_bboxes[merge_key] = [semantic.class_id, xmin, ymin, xmax, ymax]
                        
        return list(merged_bboxes.values())

    def _export_frame(self, frame_idx: int, bboxes: List[List[float]], res_w: int, res_h: int) -> None:
        """Writes YOLO labels and executes rendering passes to save images to disk."""
        label_path = self.output_dir / "labels" / f"frame_{frame_idx:05d}.txt"
        YOLOWriter.export(str(label_path), bboxes, res_w, res_h)

        rgb_pixels = self.renderer.capture_fbo_frame(self.scene, res_w, res_h, mode="RGB")
        mask_pixels = self.renderer.capture_fbo_frame(self.scene, res_w, res_h, mode="MASK")
        depth_pixels = self.renderer.capture_fbo_frame(self.scene, res_w, res_h, mode="DEPTH")
        
        img_path = self.output_dir / "images" / f"frame_{frame_idx:05d}.jpg"
        mask_path = self.output_dir / "masks" / f"frame_{frame_idx:05d}.png"
        depth_path = self.output_dir / "depth" / f"frame_{frame_idx:05d}.png"
        
        if rgb_pixels is not None and len(rgb_pixels) > 0:
            ImageWriter.save_rgb(str(img_path), rgb_pixels, res_w, res_h)
            
        if mask_pixels is not None and len(mask_pixels) > 0:
            ImageWriter.save_mask(str(mask_path), mask_pixels, res_w, res_h)
        
        if depth_pixels is not None and len(depth_pixels) > 0:
            depth_arr = np.frombuffer(depth_pixels, dtype=np.float32)
            ImageWriter.save_depth(str(depth_path), depth_arr, res_w, res_h)

    def _process_ui_events(self) -> None:
        """Keeps the Qt UI responsive during long-running generation loops."""
        try:
            from src.app import ctx
            if hasattr(ctx, 'main_window') and hasattr(ctx.main_window, 'gl_widget'):
                ctx.main_window.gl_widget.makeCurrent()
            from PySide6.QtWidgets import QApplication
            QApplication.processEvents()
        except ImportError:
            pass