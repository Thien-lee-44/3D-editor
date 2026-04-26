import random
import glm
import numpy as np
import cv2
from pathlib import Path
from typing import Any, Callable, Optional, Dict, List, Tuple
from OpenGL.GL import glClearColor, glGetFloatv, GL_COLOR_CLEAR_VALUE

from src.engine.synthetic.exporters.image_writer import ImageWriter
from src.engine.synthetic.exporters.yolo_writer import YOLOWriter
from src.engine.synthetic.exporters.coco_writer import COCOWriter
from src.engine.synthetic.exporters.metadata_writer import MetadataWriter
from src.engine.synthetic.label_utils import LabelUtils

from src.engine.scene.components import TransformComponent, MeshRenderer, CameraComponent
from src.engine.scene.components.semantic_cmp import SemanticComponent


class SyntheticDataGenerator:
    """
    Core engine responsible for rendering and exporting synthetic datasets.
    Handles automated instance tracking, domain randomization, logical masking, and bbox generation.
    """
    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.scene = engine.scene
        self.renderer = engine.renderer
        self.animator = getattr(engine, "animator", None)

        self.is_running: bool = False
        self.output_dir: Path = Path.cwd() / "datasets" / "synthetic_output"

        self.coco_writer: Optional[COCOWriter] = None
        self.metadata_writer: Optional[MetadataWriter] = None

    def setup_directories(self, custom_path: Optional[str] = None) -> None:
        if custom_path and str(custom_path).strip():
            self.output_dir = Path(custom_path)
        else:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path.cwd() / "datasets" / f"export_{timestamp}"

        for subdir in ["images", "labels", "masks/semantic", "masks/instance", "masks/instance_raw", "depth", "depth_raw", "annotations", "metadata"]:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)

        self._write_dataset_yaml()

    def _write_dataset_yaml(self) -> None:
        yaml_path = self.output_dir / "dataset.yaml"
        classes = self._get_semantic_classes()

        with open(yaml_path, "w", encoding="utf-8") as f:
            abs_dir = str(self.output_dir.absolute()).replace('\\', '/')
            f.write(f"path: {abs_dir}\n")
            f.write("train: images\n")
            f.write("val: images\n")
            f.write("annotations: annotations/instances_coco.json\n")
            f.write("metadata: metadata/frames.json\n\n")
            f.write("names:\n")

            for c_id, c_info in sorted(classes.items(), key=lambda item: int(item[0])):
                name = c_info.get("name", "Unknown") if isinstance(c_info, dict) else c_info
                f.write(f"  {int(c_id)}: {name}\n")

    def generate_batch(
        self,
        num_frames: int,
        dt: float,
        res_w: int = 1024,
        res_h: int = 1024,
        use_randomization: bool = True,
        progress_cb: Optional[Callable[[int, Optional[Dict[str, Any]], Optional[Dict[str, Any]]], None]] = None,
        preview_stride: int = 0,
        use_occlusion_bbox: bool = True,
    ) -> None:
        if not self.scene or not self.renderer:
            raise RuntimeError("Cannot generate dataset: Core subsystems are missing.")

        active_camera, active_cam_tf = self._get_active_camera()
        if not active_camera or not active_cam_tf:
            raise RuntimeError("Cannot generate dataset: No active camera found.")

        self._initialize_export_writers()
        self.is_running = True
        
        old_aspect = self._force_camera_aspect(active_camera, res_w, res_h)
        original_light_states = self._backup_lighting_state()
        original_bg_color = glGetFloatv(GL_COLOR_CLEAR_VALUE)

        try:
            current_sim_time = 0.0

            for frame_idx in range(num_frames):
                if not self.is_running:
                    break

                if self.animator:
                    self.animator.evaluate(current_sim_time, dt)

                pre_jitter_pos = glm.vec3(active_cam_tf.position)
                pre_jitter_rot = glm.quat(active_cam_tf.quat_rot)

                if use_randomization:
                    self._apply_domain_randomization(original_light_states, active_cam_tf)

                frame_payload = self._export_frame(
                    frame_idx=frame_idx,
                    sim_time=current_sim_time,
                    res_w=res_w,
                    res_h=res_h,
                    active_camera=active_camera,
                    active_cam_tf=active_cam_tf,
                    use_occlusion_bbox=use_occlusion_bbox,
                )

                if use_randomization:
                    active_cam_tf.position = pre_jitter_pos
                    active_cam_tf.quat_rot = pre_jitter_rot

                current_sim_time += dt

                if frame_payload is None:
                    if progress_cb:
                        try: progress_cb(frame_idx + 1, None, None)
                        except TypeError: pass
                    continue

                if progress_cb:
                    preview_payload = None
                    if preview_stride > 0:
                        should_preview = (
                            frame_idx == 0
                            or ((frame_idx + 1) % preview_stride == 0)
                            or frame_idx == (num_frames - 1)
                        )
                        if should_preview:
                            preview_payload = self._build_preview_payload(
                                res_w=res_w, res_h=res_h,
                                frame_data=frame_payload,
                                near=float(active_camera.near), far=float(active_camera.far),
                                is_preview=False
                            )

                    try:
                        progress_cb(frame_idx + 1, preview_payload, frame_payload["stats"])
                    except TypeError:
                        pass

        finally:
            self.is_running = False
            self._flush_export_writers()
            self._restore_lighting_state(original_light_states)
            self._restore_camera_aspect(active_camera, old_aspect)
            glClearColor(original_bg_color[0], original_bg_color[1], original_bg_color[2], original_bg_color[3])

    def preview_frame(self, res_w: int = 640, res_h: int = 640, use_randomization: bool = True, seed: Optional[int] = None, use_occlusion_bbox: bool = True, active_mode: str = "ALL", is_preview: bool = False) -> Dict[str, Any]:
        if not self.scene or not self.renderer:
            raise RuntimeError("Cannot preview dataset: Core subsystems are missing.")

        active_camera, active_cam_tf = self._get_active_camera()
        if not active_camera or not active_cam_tf:
            raise RuntimeError("Cannot preview dataset: No active camera found in the scene.")

        old_aspect = self._force_camera_aspect(active_camera, res_w, res_h)
        original_light_states = self._backup_lighting_state()
        original_bg_color = glGetFloatv(GL_COLOR_CLEAR_VALUE)

        py_random_state = random.getstate()
        np_random_state = np.random.get_state()

        try:
            if seed is not None and seed >= 0:
                random.seed(seed)
                np.random.seed(seed)

            pre_jitter_pos = glm.vec3(active_cam_tf.position)
            pre_jitter_rot = glm.quat(active_cam_tf.quat_rot)
            
            if use_randomization:
                self._apply_domain_randomization(original_light_states, active_cam_tf)

            frame_data = self._capture_frame_data(
                res_w=res_w, res_h=res_h, active_camera=active_camera, 
                use_occlusion_bbox=use_occlusion_bbox, active_mode=active_mode, 
                is_preview=is_preview
            )

            if use_randomization:
                active_cam_tf.position = pre_jitter_pos
                active_cam_tf.quat_rot = pre_jitter_rot

            return self._build_preview_payload(
                res_w=res_w, res_h=res_h,
                frame_data=frame_data,
                near=float(active_camera.near), far=float(active_camera.far),
                is_preview=is_preview
            )
        finally:
            random.setstate(py_random_state)
            np.random.set_state(np_random_state)
            self._restore_lighting_state(original_light_states)
            self._restore_camera_aspect(active_camera, old_aspect)
            glClearColor(original_bg_color[0], original_bg_color[1], original_bg_color[2], original_bg_color[3])

    def _apply_domain_randomization(self, original_states: List[Dict[str, Any]], cam_tf: TransformComponent) -> None:
        bg_type = random.randint(0, 2)
        if bg_type == 0: 
            glClearColor(random.uniform(0.3, 0.6), random.uniform(0.6, 0.8), random.uniform(0.8, 1.0), 1.0)
        elif bg_type == 1:
            glClearColor(random.uniform(0.4, 0.6), random.uniform(0.4, 0.6), random.uniform(0.4, 0.6), 1.0)
        else:
            glClearColor(random.uniform(0.8, 1.0), random.uniform(0.4, 0.6), random.uniform(0.2, 0.4), 1.0)
        
        for state in original_states:
            l_comp, l_tf = state["comp"], state["tf"]
            if l_comp.type == "Directional":
                yaw = random.uniform(0.0, 360.0)
                pitch = random.uniform(-85.0, -15.0) 
                l_tf.quat_rot = glm.angleAxis(glm.radians(yaw), glm.vec3(0, 1, 0)) * glm.angleAxis(glm.radians(pitch), glm.vec3(1, 0, 0))

                temp_mix = random.uniform(0.0, 1.0)
                warm = glm.vec3(1.0, 0.9, 0.7)
                cool = glm.vec3(0.7, 0.8, 1.0)
                base_color = warm * (1.0 - temp_mix) + cool * temp_mix
                
                intensity = random.uniform(0.7, 1.6)
                new_color = base_color * intensity
                
                if getattr(l_comp, "use_advanced_mode", False):
                    l_comp.explicit_diffuse = new_color
                    l_comp.explicit_ambient = new_color * 0.25
                else:
                    l_comp.color = new_color

        cam_tf.position += glm.vec3(random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1))
        
        pitch_jitter = random.uniform(-3.0, 3.0)
        yaw_jitter = random.uniform(-3.0, 3.0)
        roll_jitter = random.uniform(-1.5, 1.5)
        jitter_quat = glm.quat(glm.radians(glm.vec3(pitch_jitter, yaw_jitter, roll_jitter)))
        cam_tf.quat_rot = cam_tf.quat_rot * jitter_quat

    def _initialize_export_writers(self) -> None:
        classes = self._get_semantic_classes()
        self.coco_writer = COCOWriter(self.output_dir / "annotations" / "instances_coco.json", classes)
        self.metadata_writer = MetadataWriter(self.output_dir / "metadata")

    def _flush_export_writers(self) -> None:
        if self.coco_writer:
            self.coco_writer.flush()
        if self.metadata_writer:
            self.metadata_writer.flush()

    def _get_semantic_classes(self) -> Dict[int, Dict[str, Any]]:
        raw_classes: Any = {}
        if hasattr(self.engine, "get_semantic_classes"):
            raw_classes = self.engine.get_semantic_classes() or {}
        elif hasattr(self.engine, "semantic_classes"):
            raw_classes = self.engine.semantic_classes or {}

        normalized: Dict[int, Dict[str, Any]] = {}
        for raw_id, raw_info in raw_classes.items():
            class_id = int(raw_id)
            if isinstance(raw_info, dict):
                normalized[class_id] = {
                    "name": str(raw_info.get("name", f"class_{class_id}")),
                    "color": raw_info.get("color", [1.0, 1.0, 1.0]),
                }
            else:
                normalized[class_id] = {"name": str(raw_info), "color": [1.0, 1.0, 1.0]}

        if not normalized:
            normalized = {0: {"name": "Background", "color": [0.0, 0.0, 0.0]}}
        return normalized

    def _get_active_camera(self) -> Tuple[Optional[CameraComponent], Optional[TransformComponent]]:
        for tf, cam, ent in self.scene.cached_cameras:
            if getattr(cam, "is_active", False):
                return cam, tf
        return None, None

    def _force_camera_aspect(self, cam: CameraComponent, w: int, h: int) -> float:
        old_aspect = getattr(cam, "aspect_ratio", getattr(cam, "aspect", 1.0))
        target_aspect = w / h
        if hasattr(cam, "aspect_ratio"):
            cam.aspect_ratio = target_aspect
        else:
            cam.aspect = target_aspect
        if hasattr(cam, "update_projection_matrix"):
            cam.update_projection_matrix()
        return old_aspect

    def _restore_camera_aspect(self, cam: CameraComponent, old_aspect: float) -> None:
        if hasattr(cam, "aspect_ratio"):
            cam.aspect_ratio = old_aspect
        else:
            cam.aspect = old_aspect
        if hasattr(cam, "update_projection_matrix"):
            cam.update_projection_matrix()

    def _backup_lighting_state(self) -> List[Dict[str, Any]]:
        states = []
        for l_tf, light, ent in self.scene.cached_lights:
            states.append({
                "comp": light, "tf": l_tf, "color": glm.vec3(light.color),
                "explicit_diffuse": glm.vec3(light.explicit_diffuse),
                "explicit_ambient": glm.vec3(light.explicit_ambient),
                "quat_rot": glm.quat(l_tf.quat_rot), "rotation": glm.vec3(l_tf.rotation),
            })
        return states

    def _restore_lighting_state(self, states: List[Dict[str, Any]]) -> None:
        for state in states:
            l_comp, l_tf = state["comp"], state["tf"]
            l_comp.color = state["color"]
            l_comp.explicit_diffuse = state["explicit_diffuse"]
            l_comp.explicit_ambient = state["explicit_ambient"]
            l_tf.quat_rot = state["quat_rot"]
            l_tf.rotation = state["rotation"]
            if hasattr(l_tf, "sync_from_gui"):
                l_tf.sync_from_gui()

    def _resolve_leaf_track_id(self, semantic: SemanticComponent, entity_index: int) -> int:
        track_id = int(getattr(semantic, "track_id", -1))
        if track_id <= 0: 
            track_id = entity_index + 1
        return track_id

    def _collect_track_map(self) -> Dict[int, Dict[str, Any]]:
        """
        Dynamically builds a mapping of leaf tracking identities (encoded mask colors) 
        to their logical instance boundaries (root objects). 
        Ignores organizational folders and evaluates pure logical boundaries.
        """
        track_map: Dict[int, Dict[str, Any]] = {}
        for ent_idx, ent in enumerate(self.scene.entities):
            semantic = ent.get_component(SemanticComponent)
            tf = ent.get_component(TransformComponent)
            mesh = ent.get_component(MeshRenderer)

            if not semantic or not tf or not mesh: 
                continue
            if not mesh.visible or getattr(mesh, "is_proxy", False): 
                continue

            leaf_track_id = self._resolve_leaf_track_id(semantic, ent_idx)

            instance_root = ent
            curr = ent
            
            while curr.parent is not None:
                parent_sem = curr.parent.get_component(SemanticComponent)
                if parent_sem and not getattr(parent_sem, "is_merged_instance", True):
                    break
                curr = curr.parent
                instance_root = curr

            root_sem = instance_root.get_component(SemanticComponent)
            if not root_sem:
                continue

            logical_track_id = int(getattr(root_sem, "track_id", -1))
            if logical_track_id <= 0:
                root_idx = self.scene.entities.index(instance_root)
                logical_track_id = root_idx + 1  

            class_id = int(getattr(root_sem, "class_id", getattr(semantic, "class_id", 0)))
            root_name = str(getattr(instance_root, "name", f"entity_{ent_idx}"))
            
            if not hasattr(instance_root, "parent") and '/' in root_name:
                root_name = root_name.split('/')[0]

            if leaf_track_id not in track_map:
                world_pos = getattr(tf, "global_position", tf.position)
                track_map[leaf_track_id] = {
                    "track_id": logical_track_id, 
                    "class_id": class_id, 
                    "entity_name": root_name,
                    "root_id": id(instance_root), 
                    "world_position": self._vec3_to_list(world_pos),
                }
                
        return track_map

    def _map_to_logical_mask(self, instance_mask_pixels: bytes, res_w: int, res_h: int, track_map: Dict[int, Dict[str, Any]]) -> np.ndarray:
        """Translates raw FBO leaf pixel identities into unified logical object boundaries."""
        if not instance_mask_pixels:
            return np.zeros((res_h, res_w), dtype=np.uint32)

        pixel_arr = np.frombuffer(instance_mask_pixels, dtype=np.uint8).reshape((res_h, res_w, 3))

        ids = (pixel_arr[:, :, 0].astype(np.uint32) | 
               (pixel_arr[:, :, 1].astype(np.uint32) << 8) | 
               (pixel_arr[:, :, 2].astype(np.uint32) << 16))
               
        logical_ids = np.zeros_like(ids)
        unique_ids = np.unique(ids)
        
        for leaf_id in unique_ids:
            if leaf_id == 0 or leaf_id == 16777215:
                continue
            
            leaf_id_int = int(leaf_id)
            if track_map and leaf_id_int in track_map:
                logical_id = track_map[leaf_id_int]["track_id"]
                logical_ids[ids == leaf_id] = logical_id
            else:
                logical_ids[ids == leaf_id] = leaf_id_int
                
        return logical_ids

    def _extract_visible_objects_from_instance_mask(self, logical_mask: np.ndarray, track_map: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parses the unified logical mask to extract precise bounding boxes and polygon segmentations."""
        logical_mask_td = np.flipud(logical_mask)
        present_ids = np.unique(logical_mask_td)
        
        logical_to_info = {}
        for leaf_id, info in track_map.items():
            logical_to_info[info["track_id"]] = info
            
        visible_objects: List[Dict[str, Any]] = []
        for logical_id in present_ids:
            if logical_id == 0 or logical_id == 16777215: continue
            if logical_id not in logical_to_info: continue
            
            group_mask = (logical_mask_td == logical_id).astype(np.uint8) * 255
            ys, xs = np.where(group_mask > 0)
            if xs.size == 0: continue

            xmin, ymin = float(xs.min()), float(ys.min())
            xmax, ymax = float(xs.max() + 1), float(ys.max() + 1)
            visible_pixels = int(xs.size)

            segmentation = []
            contours, _ = cv2.findContours(group_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if contour.shape[0] >= 3:
                    poly = contour.flatten().tolist()
                    segmentation.append([float(v) for v in poly])

            track_info = logical_to_info[int(logical_id)]
            bbox_area = max((xmax - xmin) * (ymax - ymin), 1.0)
            vis_ratio = min(1.0, visible_pixels / bbox_area)
            
            visible_objects.append({
                "track_id": track_info["track_id"], "class_id": track_info["class_id"],
                "entity_name": track_info["entity_name"], "world_position": track_info["world_position"],
                "bbox_xyxy": [xmin, ymin, xmax, ymax], "segmentation": segmentation, 
                "visible_pixels": visible_pixels, "visibility_ratio": float(vis_ratio), "occlusion_ratio": float(1.0 - vis_ratio),
            })

        return visible_objects

    def _calculate_bounding_boxes(self, view_mat: glm.mat4, proj_mat: glm.mat4, res_w: int, res_h: int) -> List[Tuple[int, float, float, float, float, str, int]]:
        """Fallback un-occluded bounding box calculation."""
        merged_bboxes = {}
        for ent in self.scene.entities:
            semantic = ent.get_component(SemanticComponent)
            tf = ent.get_component(TransformComponent)
            mesh = ent.get_component(MeshRenderer)

            if semantic and tf and mesh and mesh.visible and not getattr(mesh, "is_proxy", False):
                bbox = LabelUtils.get_2d_bounding_box(tf, mesh.geometry, view_mat, proj_mat, res_w, res_h)
                if bbox:
                    xmin, ymin, xmax, ymax = bbox
                    curr = ent
                    while hasattr(curr, "parent") and curr.parent is not None:
                        parent_sem = curr.parent.get_component(SemanticComponent)
                        if parent_sem and not getattr(parent_sem, "is_merged_instance", True):
                            break
                        curr = curr.parent
                        
                    root_name = str(getattr(curr, "name", "UnknownGroup"))
                    if not hasattr(curr, "parent") and '/' in root_name:
                        root_name = root_name.split('/')[0]
                        
                    root_id = id(curr)
                    root_sem = curr.get_component(SemanticComponent) if hasattr(curr, "get_component") else None
                    class_id = int(getattr(root_sem, "class_id", getattr(semantic, "class_id", 0)))

                    logical_track_id = int(getattr(root_sem, "track_id", -1))
                    if logical_track_id <= 0:
                        root_idx = self.scene.entities.index(curr)
                        logical_track_id = root_idx + 1

                    if root_id in merged_bboxes:
                        curr_box = merged_bboxes[root_id]
                        merged_bboxes[root_id] = (
                            class_id, min(curr_box[1], xmin), min(curr_box[2], ymin),
                            max(curr_box[3], xmax), max(curr_box[4], ymax), root_name, logical_track_id
                        )
                    else:
                        merged_bboxes[root_id] = (class_id, xmin, ymin, xmax, ymax, root_name, logical_track_id)

        return list(merged_bboxes.values())

    def _colorize_instance_mask(self, logical_mask: np.ndarray) -> bytes:
        """Applies highly distinct randomized colors seeded by Logical IDs for visual debugging."""
        if logical_mask.size == 0: return b""
              
        colorized = np.zeros((*logical_mask.shape, 3), dtype=np.uint8)
        unique_ids = np.unique(logical_mask)
        
        for logical_id in unique_ids:
            if logical_id == 0 or logical_id == 16777215: 
                continue
                
            rng = np.random.default_rng(int(logical_id))
            color = rng.integers(50, 255, size=3, dtype=np.uint8)
            colorized[logical_mask == logical_id] = color
            
        return colorized.tobytes()

    def _capture_frame_data(self, res_w: int, res_h: int, active_camera: CameraComponent, use_occlusion_bbox: bool, active_mode: str = "ALL", is_preview: bool = False) -> Dict[str, Any]:
        texture_id = 0
        rgb_pixels = b""
        semantic_mask_pixels = b""
        instance_mask_pixels = b""
        logical_mask = np.zeros(0, dtype=np.uint32)
        depth_arr = np.zeros(0, dtype=np.float32)

        track_map = self._collect_track_map()

        if is_preview:
            if active_mode == "SEMANTIC": target_mode = "MASK_SEMANTIC"
            elif active_mode == "INSTANCE": target_mode = "MASK_INSTANCE"
            else: target_mode = active_mode if active_mode != "ALL" else "RGB"

            pixel_data = self.renderer.capture_fbo_frame(self.scene, res_w, res_h, mode=target_mode, return_texture_id=False)

            if target_mode == "RGB": 
                rgb_pixels = pixel_data
            elif target_mode == "MASK_SEMANTIC": 
                semantic_mask_pixels = pixel_data
            elif target_mode == "MASK_INSTANCE": 
                logical_mask = self._map_to_logical_mask(pixel_data, res_w, res_h, track_map)
                instance_mask_pixels = self._colorize_instance_mask(logical_mask)
            elif target_mode == "DEPTH":
                if pixel_data:
                    depth_arr = np.frombuffer(pixel_data, dtype=np.float32)

            if use_occlusion_bbox and target_mode != "MASK_INSTANCE":
                raw_data = self.renderer.capture_fbo_frame(self.scene, res_w, res_h, mode="MASK_INSTANCE", return_texture_id=False)
                logical_mask = self._map_to_logical_mask(raw_data, res_w, res_h, track_map)
        else:
            depth_arr = np.full((res_w * res_h,), np.inf, dtype=np.float32)
            if active_mode in ["ALL", "RGB"]:
                rgb_pixels = self.renderer.capture_fbo_frame(self.scene, res_w, res_h, mode="RGB")
            if active_mode in ["ALL", "SEMANTIC"]:
                semantic_mask_pixels = self.renderer.capture_fbo_frame(self.scene, res_w, res_h, mode="MASK_SEMANTIC")
            if active_mode in ["ALL", "INSTANCE"] or use_occlusion_bbox:
                raw_data = self.renderer.capture_fbo_frame(self.scene, res_w, res_h, mode="MASK_INSTANCE")
                logical_mask = self._map_to_logical_mask(raw_data, res_w, res_h, track_map)
                if active_mode in ["ALL", "INSTANCE"]:
                    instance_mask_pixels = self._colorize_instance_mask(logical_mask)
            if active_mode in ["ALL", "DEPTH"]:
                depth_pixels = self.renderer.capture_fbo_frame(self.scene, res_w, res_h, mode="DEPTH")
                if depth_pixels is not None and len(depth_pixels) > 0:
                    depth_arr = np.frombuffer(depth_pixels, dtype=np.float32)

        visible_objects: List[Dict[str, Any]] = []
        
        if use_occlusion_bbox and logical_mask.size > 0:
            visible_objects = self._extract_visible_objects_from_instance_mask(logical_mask, track_map)

        if not visible_objects:
            view_mat = active_camera.get_view_matrix()
            proj_mat = active_camera.get_projection_matrix()
            projected_bboxes = self._calculate_bounding_boxes(view_mat, proj_mat, res_w, res_h)

            for idx, (class_id, xmin, ymin, xmax, ymax, group_name, logical_track_id) in enumerate(projected_bboxes):
                area = max((xmax - xmin) * (ymax - ymin), 1.0)
                visible_objects.append({
                    "track_id": logical_track_id, "class_id": int(class_id), "entity_name": group_name,
                    "world_position": [0.0, 0.0, 0.0], "bbox_xyxy": [float(xmin), float(ymin), float(xmax), float(ymax)],
                    "segmentation": [], "visible_pixels": int(area), "visibility_ratio": 1.0, "occlusion_ratio": 0.0,
                })

        return {
            "texture_id": texture_id,
            "rgb_pixels": rgb_pixels, 
            "semantic_mask_pixels": semantic_mask_pixels,
            "instance_mask_pixels": instance_mask_pixels, 
            "logical_mask_arr": logical_mask, 
            "depth_arr": depth_arr,
            "visible_objects": visible_objects, 
            "stats": self._summarize_objects(visible_objects),
        }

    def _depth_to_preview_rgb(self, depth_arr: np.ndarray, res_w: int, res_h: int, near: float, far: float) -> bytes:
        depth_map = np.array(depth_arr, dtype=np.float32).reshape((res_h, res_w))
        valid = np.isfinite(depth_map) & (depth_map > 0.0)
        out = np.zeros((res_h, res_w), dtype=np.uint8)

        if np.any(valid):
            d_min, d_max = near, far
            if d_max <= d_min:
                valid_values = depth_map[valid]
                d_min = float(np.percentile(valid_values, 1.0))
                d_max = float(np.percentile(valid_values, 99.0))
            if d_max > d_min:
                clamped = np.clip(depth_map, d_min, d_max)
                normalized = (clamped - d_min) / (d_max - d_min)
                out[valid] = (normalized[valid] * 255.0).astype(np.uint8)

        rgb = np.stack([out, out, out], axis=-1)
        return rgb.tobytes()

    def _summarize_objects(self, visible_objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        classes = self._get_semantic_classes()
        class_hist: Dict[str, int] = {}
        for obj in visible_objects:
            class_id = int(obj.get("class_id", 0))
            class_info = classes.get(class_id, {"name": "Unknown"})
            class_name = class_info.get("name", "Unknown") if isinstance(class_info, dict) else str(class_info)
            class_hist[class_name] = class_hist.get(class_name, 0) + 1

        mean_occ = float(sum(float(o.get("occlusion_ratio", 0.0)) for o in visible_objects) / len(visible_objects)) if visible_objects else 0.0
        return {"num_objects": len(visible_objects), "mean_occlusion": mean_occ, "class_hist": class_hist}

    def _build_preview_payload(self, res_w: int, res_h: int, frame_data: Dict[str, Any], near: float, far: float, is_preview: bool = False) -> Dict[str, Any]:
        depth_preview_rgb = b""
        depth_arr = frame_data.get("depth_arr")
        
        if depth_arr is not None and getattr(depth_arr, 'size', 0) > 0:
            depth_preview_rgb = self._depth_to_preview_rgb(depth_arr, res_w, res_h, near, far)

        return {
            "width": int(res_w), "height": int(res_h),
            "is_gpu_direct": False,
            "texture_id": 0,
            "modes": {
                "RGB": frame_data.get("rgb_pixels", b""), 
                "SEMANTIC": frame_data.get("semantic_mask_pixels", b""), 
                "INSTANCE": frame_data.get("instance_mask_pixels", b""), 
                "DEPTH": depth_preview_rgb
            },
            "stats": frame_data.get("stats", {}), 
            "objects": frame_data.get("visible_objects", []),
        }

    def _export_frame(self, frame_idx: int, sim_time: float, res_w: int, res_h: int, active_camera: CameraComponent, active_cam_tf: TransformComponent, use_occlusion_bbox: bool = True) -> Optional[Dict[str, Any]]:
        frame_data = self._capture_frame_data(res_w=res_w, res_h=res_h, active_camera=active_camera, use_occlusion_bbox=use_occlusion_bbox)
        visible_objects = frame_data["visible_objects"]

        img_rel = f"images/frame_{frame_idx:05d}.jpg"
        yolo_rel = f"labels/frame_{frame_idx:05d}.txt"
        sem_mask_rel = f"masks/semantic/frame_{frame_idx:05d}.png"
        inst_mask_rel = f"masks/instance/frame_{frame_idx:05d}.png"
        inst_raw_rel = f"masks/instance_raw/frame_{frame_idx:05d}.npy"
        depth_rel = f"depth/frame_{frame_idx:05d}.png"
        depth_raw_rel = f"depth_raw/frame_{frame_idx:05d}.npy"

        img_path = self.output_dir / img_rel
        yolo_path = self.output_dir / yolo_rel
        
        rgb_pixels = frame_data["rgb_pixels"]
        if rgb_pixels: 
            ImageWriter.save_rgb(str(img_path), rgb_pixels, res_w, res_h)
        
        YOLOWriter.export(str(yolo_path), visible_objects, res_w, res_h, is_segmentation=use_occlusion_bbox)

        if frame_data["semantic_mask_pixels"]: 
            ImageWriter.save_mask(str(self.output_dir / sem_mask_rel), frame_data["semantic_mask_pixels"], res_w, res_h)
        if frame_data["instance_mask_pixels"]: 
            ImageWriter.save_mask(str(self.output_dir / inst_mask_rel), frame_data["instance_mask_pixels"], res_w, res_h)

        if frame_data.get("logical_mask_arr") is not None and frame_data["logical_mask_arr"].size > 0:
            logical_mask_td = np.flipud(frame_data["logical_mask_arr"])
            np.save(str(self.output_dir / inst_raw_rel), logical_mask_td)

        if visible_objects and self.coco_writer: 
            self.coco_writer.add_frame(frame_idx, img_rel, res_w, res_h, visible_objects)

        depth_arr = frame_data["depth_arr"]
        if depth_arr is not None and getattr(depth_arr, 'size', 0) > 0:
            ImageWriter.save_depth(str(self.output_dir / depth_rel), depth_arr, res_w, res_h, near=float(active_camera.near), far=float(active_camera.far))
            ImageWriter.save_depth_npy(str(self.output_dir / depth_raw_rel), depth_arr, res_w, res_h)

        if self.metadata_writer:
            files_dict = {
                "image": img_rel, 
                "label": yolo_rel, 
                "semantic_mask": sem_mask_rel, 
                "instance_mask": inst_mask_rel, 
                "instance_raw": inst_raw_rel,
                "depth_png": depth_rel, 
                "depth_npy": depth_raw_rel
            }
            frame_record = self._build_frame_metadata(
                frame_idx=frame_idx, sim_time=sim_time,
                files=files_dict,
                active_camera=active_camera, active_cam_tf=active_cam_tf, visible_objects=visible_objects, stats=frame_data["stats"]
            )
            self.metadata_writer.add_frame(frame_record)

        return {
            "rgb_pixels": rgb_pixels, "semantic_mask_pixels": frame_data["semantic_mask_pixels"],
            "instance_mask_pixels": frame_data["instance_mask_pixels"], "depth_arr": depth_arr,
            "visible_objects": visible_objects, "stats": frame_data["stats"],
        }

    def _build_frame_metadata(self, frame_idx: int, sim_time: float, files: Dict[str, str], active_camera: CameraComponent, active_cam_tf: TransformComponent, visible_objects: List[Dict[str, Any]], stats: Dict[str, Any]) -> Dict[str, Any]:
        classes = self._get_semantic_classes()
        camera_pos = getattr(active_cam_tf, "global_position", active_cam_tf.position)
        camera_quat = getattr(active_cam_tf, "global_quat_rot", getattr(active_cam_tf, "quat_rot", None))

        objects_payload: List[Dict[str, Any]] = []
        for obj in visible_objects:
            class_id = int(obj.get("class_id", 0))
            class_info = classes.get(class_id, {"name": "Unknown"})
            class_name = class_info.get("name", "Unknown") if isinstance(class_info, dict) else str(class_info)
            objects_payload.append({
                "track_id": int(obj.get("track_id", -1)), "class_id": class_id, "class_name": class_name,
                "entity_name": obj.get("entity_name", "unknown"), "world_position": obj.get("world_position", [0.0, 0.0, 0.0]),
                "bbox_xyxy": obj.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0]), "visible_pixels": int(obj.get("visible_pixels", 0)),
                "visibility_ratio": float(obj.get("visibility_ratio", 0.0)), "occlusion_ratio": float(obj.get("occlusion_ratio", 0.0)),
            })

        class_hist = stats.get("class_hist", {})
        if class_hist:
            items = [f"{count} {name.lower()}(s)" for name, count in class_hist.items()]
            caption = f"A synthetic driving scene containing {', '.join(items)}."
        else:
            caption = "An empty synthetic driving scene."

        return {
            "frame_index": int(frame_idx), "sim_time": float(sim_time), "caption": caption, "files": files,
            "camera": {
                "mode": str(getattr(active_camera, "mode", "Perspective")), "fov": float(getattr(active_camera, "fov", 45.0)),
                "near": float(getattr(active_camera, "near", 0.1)), "far": float(getattr(active_camera, "far", 1000.0)),
                "position": self._vec3_to_list(camera_pos), "rotation": self._vec3_to_list(getattr(active_cam_tf, "rotation", glm.vec3(0.0))),
                "quat": self._quat_to_list(camera_quat),
            },
            "objects": objects_payload,
        }

    def _vec3_to_list(self, value: Any) -> List[float]:
        try: return [float(value.x), float(value.y), float(value.z)]
        except Exception: pass
        try: return [float(value[0]), float(value[1]), float(value[2])]
        except Exception: return [0.0, 0.0, 0.0]

    def _quat_to_list(self, value: Any) -> List[float]:
        if value is None: return [1.0, 0.0, 0.0, 0.0]
        try: return [float(value.w), float(value.x), float(value.y), float(value.z)]
        except Exception: pass
        try: return [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
        except Exception: return [1.0, 0.0, 0.0, 0.0]