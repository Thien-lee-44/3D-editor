"""
Synthetic Data Generator.

The core execution engine for generating large-scale synthetic datasets.
Coordinates the Scene, Animator, and Renderers to output synchronized frames, 
masks, and annotations (YOLO, COCO, VOC) over a continuous simulation timeline.
Outputs data in a flat directory structure, leaving Train/Val/Test splits 
to be handled dynamically by the CV Benchmark tool.
"""

import math
import shutil
import random
import json
from pathlib import Path
from typing import Any, Callable, Optional, Dict, List, Tuple, Union
from datetime import datetime
import os

import glm
import numpy as np

from src.engine.synthetic.label_utils import LabelUtils
from src.engine.scene.components import TransformComponent, CameraComponent
from src.engine.synthetic.exporters.image_writer import ImageWriter
from src.engine.synthetic.exporters.yolo_writer import YOLOWriter
from src.engine.synthetic.exporters.coco_writer import COCOWriter
from src.engine.synthetic.exporters.metadata_writer import MetadataWriter
from src.engine.synthetic.exporters.voc_writer import VOCWriter


class SyntheticDataGenerator:
    """
    Orchestrates the automated generation of photorealistic and ground-truth data.
    Features integrated domain randomization for lighting and camera viewpoints.
    """
    
    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.scene = engine.scene
        self.renderer = engine.renderer
        self.synthetic_renderer = getattr(engine, "synthetic_renderer", None)
        self.animator = getattr(engine, "animator", None)

        self.is_running: bool = False
        self.output_dir: Path = Path.cwd() / "datasets" / "synthetic_output"

    def _get_active_camera(self) -> Tuple[Optional[CameraComponent], Optional[TransformComponent]]:
        """Resolves the currently active camera used for dataset viewpoint rendering."""
        for tf, cam, ent in self.scene.cached_cameras:
            if getattr(cam, 'is_active', False):
                return cam, tf
        return None, None

    def _get_class_name_lookup(self) -> Dict[int, str]:
        """Builds a mapping of class IDs to human-readable names from the Semantic Manager."""
        classes: Dict[int, Any] = {}
        if hasattr(self.engine, 'get_semantic_classes'):
            try:
                classes = self.engine.get_semantic_classes() or {}
            except Exception:
                classes = {}

        lookup: Dict[int, str] = {}
        for raw_id, raw_info in classes.items():
            class_id = int(raw_id)
            if isinstance(raw_info, dict):
                lookup[class_id] = str(raw_info.get('name', f'class_{class_id}'))
            else:
                lookup[class_id] = str(raw_info)

        if not lookup:
            lookup = {0: 'object'}

        return lookup

    def _write_dataset_yaml(self, class_lookup: Dict[int, str]) -> None:
        """Generates a flat YOLO-compatible dataset configuration file."""
        yaml_path = self.output_dir / 'dataset.yaml'
        lines = [
            f"path: {self.output_dir.as_posix()}",
            "train: images",
            "val: images",
            f"nc: {len(class_lookup)}",
            "names:",
        ]

        for class_id, class_name in sorted(class_lookup.items(), key=lambda item: int(item[0])):
            safe_name = str(class_name).replace('"', "'")
            lines.append(f"  {int(class_id)}: \"{safe_name}\"")

        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines) + "\n")

    def _write_export_manifest(self) -> None:
        """Creates a JSON index of all generated data formats and their relative paths."""
        manifest_path = self.output_dir / "annotations" / "export_manifest.json"
        payload = {
            "dataset_root": self.output_dir.as_posix(),
            "formats": {
                "yolo_detection": "labels/*.txt",
                "yolo_segmentation": "labels_seg/*.txt",
                "coco_detection_instance": "annotations/instances_coco.json",
                "pascal_voc": "annotations/voc/*.xml",
                "metadata_json": "metadata/frames.json",
                "metadata_csv": "metadata/objects.csv",
                "metadata_ndjson": "metadata/frames.ndjson",
                "rgb": "images/*.jpg",
                "depth_png": "depth/*.png",
                "depth_npy": "depth/*.npy",
                "instance_mask": "masks/instance/*.png",
                "semantic_mask": "masks/semantic/*.png",
            },
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _camera_payload(self, cam: CameraComponent, tf: Optional[TransformComponent]) -> Dict[str, Any]:
        """Serializes camera state for metadata tracking."""
        if tf:
            pos = [float(tf.global_position.x), float(tf.global_position.y), float(tf.global_position.z)]
            rot = [float(tf.global_quat_rot.w), float(tf.global_quat_rot.x), float(tf.global_quat_rot.y), float(tf.global_quat_rot.z)]
        else:
            pos = [0.0, 0.0, 0.0]
            rot = [1.0, 0.0, 0.0, 0.0]

        return {
            'mode': str(getattr(cam, 'mode', 'Perspective')),
            'fov': float(getattr(cam, 'fov', 60.0)),
            'near': float(getattr(cam, 'near', 0.1)),
            'far': float(getattr(cam, 'far', 100.0)),
            'ortho_size': float(getattr(cam, 'ortho_size', 10.0)),
            'position': pos,
            'rotation_quat': rot,
        }

    def _extract_accurate_bboxes(self, inst_pixels: bytes, w: int, h: int, class_lookup: Dict[int, str]) -> List[Dict[str, Any]]:
        """
        Processes raw pixel masks to generate coordinate-accurate bounding boxes.
        Automatically corrects OpenGL's inverted Y-axis.
        """
        bboxes_raw = LabelUtils.extract_bboxes_from_mask(inst_pixels, w, h)
        objects_payload = []
        processed_tracks = set()
        
        for ent in self.scene.entities:
            from src.engine.scene.components.semantic_cmp import SemanticComponent
            from src.engine.scene.components import MeshRenderer
            
            sem = ent.get_component(SemanticComponent)
            mesh = ent.get_component(MeshRenderer)
            
            if sem and mesh and mesh.visible and not getattr(mesh, 'is_proxy', False):
                track_id = -1
                if hasattr(self.engine, 'get_resolved_track_id'):
                    track_id = self.engine.get_resolved_track_id(ent)
                if track_id <= 0:
                    track_id = self.scene.entities.index(ent) + 1
                    
                if track_id in bboxes_raw and track_id not in processed_tracks:
                    box_data = bboxes_raw[track_id]
                    xmin, ymin, xmax, ymax = box_data["bbox"]
                    
                    # Flip Y axis from OpenGL (bottom-left) to Image (top-left) standard
                    corrected_ymin = h - ymax
                    corrected_ymax = h - ymin
                    
                    corrected_bbox = [float(xmin), float(corrected_ymin), float(xmax), float(corrected_ymax)]
                    
                    class_name = class_lookup.get(sem.class_id, getattr(ent, 'name', 'Unknown'))
                    objects_payload.append({
                        "track_id": track_id,
                        "class_id": sem.class_id,
                        "class_name": class_name,
                        "bbox": corrected_bbox,
                        "bbox_xyxy": corrected_bbox
                    })
                    processed_tracks.add(track_id)
                    
        return objects_payload

    def extract_preview_frame(self, w: int, h: int, mode: str, is_playing: bool, show_bbox: bool = True) -> Dict[str, Any]:
        """Renders a single frame targeted for the UI Preview Viewport without saving to disk."""
        cam, tf = self._get_active_camera()
        if not cam or not self.synthetic_renderer:
            return {}

        cam.aspect = w / max(h, 1)
        active_mode = mode if mode in {'RGB', 'DEPTH', 'SEMANTIC', 'INSTANCE'} else 'RGB'
        
        modes_to_capture = []
        if active_mode == 'INSTANCE':
            modes_to_capture.append('INSTANCE_PREVIEW')
        else:
            modes_to_capture.append(active_mode)

        if show_bbox and 'INSTANCE' not in modes_to_capture:
            modes_to_capture.append('INSTANCE')

        raw_modes = self.synthetic_renderer.capture_fbo_frames(self.scene, w, h, modes_to_capture)
        class_lookup = self._get_class_name_lookup()

        payload = {
            "width": w,
            "height": h,
            "mode": active_mode,
            "is_live": bool(is_playing),
            "modes": {},
            "objects": [],
            "stats": {"num_objects": 0}
        }

        if active_mode == 'INSTANCE':
            payload["modes"]['INSTANCE'] = raw_modes.get('INSTANCE_PREVIEW', b'')
        else:
            payload["modes"][active_mode] = raw_modes.get(active_mode, b'')

        if show_bbox and 'INSTANCE' in raw_modes:
            inst_pixels = raw_modes.get('INSTANCE', b'')
            payload["objects"] = self._extract_accurate_bboxes(inst_pixels, w, h, class_lookup)

        payload["stats"]["num_objects"] = len(payload["objects"])
        return payload

    def setup_directories(self, target_dir: Optional[Union[str, Path]] = None) -> None:
        """Creates a flat directory structure for dataset outputs."""
        if target_dir:
            self.output_dir = Path(target_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path(os.getcwd()) / "datasets" / f"synth_data_{timestamp}"
            
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        for d in ["images", "labels", "labels_seg", "depth", "masks/instance", "masks/semantic", "annotations/voc"]:
            (self.output_dir / d).mkdir(parents=True, exist_ok=True)
                
        (self.output_dir / "metadata").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "annotations").mkdir(parents=True, exist_ok=True)

    def generate_batch(self, num_frames: int, dt: float, res_w: int, res_h: int, use_rand_light: bool, use_rand_cam: bool, progress_cb: Optional[Callable] = None, preview_stride: int = 1) -> None:
        """
        The Main Dataset Generation Loop.
        Iterates over the timeline, evaluates animations, applies domain randomization,
        triggers the rendering pipeline, and flushes data to flat folders.
        """
        self.is_running = True
        
        cam, cam_tf = self._get_active_camera()
        if not cam or not self.synthetic_renderer:
            self.is_running = False
            return
            
        cam.aspect = res_w / max(res_h, 1)

        class_lookup = self._get_class_name_lookup()
        classes_meta = getattr(self.engine, 'get_semantic_classes', lambda: {})() or {}
        if not classes_meta:
            classes_meta = {cid: {'name': name, 'color': [1.0, 1.0, 1.0]} for cid, name in class_lookup.items()}

        coco_writer = COCOWriter(self.output_dir / 'annotations' / 'instances_coco.json', classes_meta)
        metadata_writer = MetadataWriter(self.output_dir / 'metadata')

        sim_time = 0.0

        # Cache baseline lighting states for accurate domain randomization
        clean_lights = []
        for l_tf, l_comp, _ in self.scene.cached_lights:
            if getattr(l_comp, 'type', '') == "Directional":
                clean_lights.append({
                    'tf': l_tf,
                    'comp': l_comp,
                    'rot': glm.vec3(l_tf.rotation),
                    'color': glm.vec3(l_comp.color),
                    'intensity': float(l_comp.intensity)
                })

        try:
            for frame_idx in range(num_frames):
                if not self.is_running:
                    break

                if self.animator:
                    self.animator.evaluate(sim_time, dt, target_entity_id=-1)

                clean_cam_rot = glm.vec3(cam_tf.rotation) if cam_tf else None
                bg_color = (0.0, 0.0, 0.0)

                # Apply Camera Domain Randomization
                if use_rand_cam and cam_tf and clean_cam_rot is not None:
                    cam_tf.rotation = clean_cam_rot + glm.vec3(
                        random.uniform(-5.0, 5.0), 
                        random.uniform(-15.0, 15.0), 
                        random.uniform(-2.0, 2.0)
                    )
                    cam_tf.quat_rot = glm.quat(glm.radians(cam_tf.rotation))
                    cam_tf.is_dirty = True

                # Apply Lighting Domain Randomization (Simulated Day-Night Cycle)
                if use_rand_light:
                    time_of_day = random.uniform(6.0, 18.0)
                    t_norm = (time_of_day - 6.0) / 12.0
                    
                    sun_pitch = -90.0 * math.sin(math.pi * t_norm)
                    sun_yaw = random.uniform(0.0, 360.0)
                    
                    factor = math.sin(math.pi * t_norm)
                    
                    sun_color_dawn = glm.vec3(1.0, 0.6, 0.3)
                    sun_color_noon = glm.vec3(1.0, 0.95, 0.9)
                    sun_color = glm.vec3(
                        sun_color_dawn.x * (1.0 - factor) + sun_color_noon.x * factor,
                        sun_color_dawn.y * (1.0 - factor) + sun_color_noon.y * factor,
                        sun_color_dawn.z * (1.0 - factor) + sun_color_noon.z * factor
                    )
                    
                    sky_color_dawn = glm.vec3(0.5, 0.3, 0.2)
                    sky_color_noon = glm.vec3(0.3, 0.6, 0.9)
                    sky_color = glm.vec3(
                        sky_color_dawn.x * (1.0 - factor) + sky_color_noon.x * factor,
                        sky_color_dawn.y * (1.0 - factor) + sky_color_noon.y * factor,
                        sky_color_dawn.z * (1.0 - factor) + sky_color_noon.z * factor
                    )
                    bg_color = (sky_color.x, sky_color.y, sky_color.z)
                    intensity = 0.4 + 0.8 * factor

                    for l_data in clean_lights:
                        l_tf = l_data['tf']
                        l_comp = l_data['comp']
                        
                        l_tf.rotation = glm.vec3(sun_pitch, sun_yaw, 0.0)
                        l_tf.quat_rot = glm.quat(glm.radians(l_tf.rotation))
                        l_tf.is_dirty = True
                        
                        l_comp.color = sun_color
                        l_comp.intensity = intensity

                capture_modes = ['RGB', 'DEPTH', 'INSTANCE', 'SEMANTIC']
                mode_frames = self.synthetic_renderer.capture_fbo_frames(
                    self.scene, res_w, res_h, capture_modes, download=True, bg_color=bg_color
                )
                
                # Restore states after capture
                if use_rand_cam and cam_tf and clean_cam_rot is not None:
                    cam_tf.rotation = clean_cam_rot
                    cam_tf.quat_rot = glm.quat(glm.radians(cam_tf.rotation))
                    cam_tf.is_dirty = True
                        
                if use_rand_light:
                    for l_data in clean_lights:
                        l_tf = l_data['tf']
                        l_comp = l_data['comp']
                        
                        l_tf.rotation = l_data['rot']
                        l_tf.quat_rot = glm.quat(glm.radians(l_tf.rotation))
                        l_tf.is_dirty = True
                        
                        l_comp.color = l_data['color']
                        l_comp.intensity = l_data['intensity']

                rgb_pixels = mode_frames.get('RGB', b'')
                depth_pixels = mode_frames.get('DEPTH', b'')
                inst_pixels = mode_frames.get('INSTANCE', b'')
                sem_pixels = mode_frames.get('SEMANTIC', b'')

                objects_payload = self._extract_accurate_bboxes(inst_pixels, res_w, res_h, class_lookup)

                # =========================================================
                # FLAT DIRECTORY EXPORT
                # =========================================================
                frame_name = f"frame_{frame_idx:06d}"
                rel_rgb = f"images/{frame_name}.jpg"
                rel_depth = f"depth/{frame_name}.png"
                rel_depth_npy = f"depth/{frame_name}.npy"
                rel_mask_instance = f"masks/instance/{frame_name}.png"
                rel_mask_semantic = f"masks/semantic/{frame_name}.png"
                rel_label = f"labels/{frame_name}.txt"
                rel_label_seg = f"labels_seg/{frame_name}.txt"
                rel_voc = f"annotations/voc/{frame_name}.xml"

                # Disk I/O Serialization
                ImageWriter.save_rgb(str(self.output_dir / rel_rgb), rgb_pixels, res_w, res_h)
                ImageWriter.save_mask(str(self.output_dir / rel_mask_instance), inst_pixels, res_w, res_h)
                ImageWriter.save_mask(str(self.output_dir / rel_mask_semantic), sem_pixels, res_w, res_h)

                depth_arr = np.frombuffer(depth_pixels, dtype=np.uint8)
                ImageWriter.save_depth(str(self.output_dir / rel_depth), depth_arr, res_w, res_h, near=0.0, far=255.0)
                ImageWriter.save_depth_npy(str(self.output_dir / rel_depth_npy), depth_arr, res_w, res_h)

                YOLOWriter.export(str(self.output_dir / rel_label), objects_payload, res_w, res_h, is_segmentation=False)
                YOLOWriter.export(str(self.output_dir / rel_label_seg), objects_payload, res_w, res_h, is_segmentation=True)
                coco_writer.add_frame(frame_idx, rel_rgb, res_w, res_h, objects_payload)
                VOCWriter.export(str(self.output_dir / rel_voc), rel_rgb, res_w, res_h, objects_payload)

                frame_record = {
                    'frame_index': frame_idx,
                    'time_s': float(sim_time),
                    'files': {
                        'image': rel_rgb,
                        'depth': rel_depth,
                        'depth_npy': rel_depth_npy,
                        'instance_mask': rel_mask_instance,
                        'semantic_mask': rel_mask_semantic,
                        'label': rel_label,
                        'label_seg': rel_label_seg,
                        'voc': rel_voc,
                    },
                    'camera': self._camera_payload(cam, cam_tf),
                    'objects': objects_payload,
                    'stats': {'num_objects': len(objects_payload)},
                }
                metadata_writer.add_frame(frame_record)

                preview_payload = None
                if frame_idx % max(1, preview_stride) == 0:
                    preview_payload = {
                        'width': res_w,
                        'height': res_h,
                        'is_live': False,
                        'modes': {
                            'RGB': rgb_pixels,
                            'DEPTH': depth_pixels,
                            'INSTANCE': inst_pixels,
                            'SEMANTIC': sem_pixels
                        },
                        'mode': 'RGB',
                        'objects': objects_payload,
                        'stats': frame_record['stats'],
                    }

                if progress_cb:
                    progress_cb(frame_idx + 1, preview_payload, frame_record['stats'])

                sim_time += dt

        finally:
            coco_writer.flush()
            metadata_writer.flush()
            self._write_dataset_yaml(class_lookup)
            self._write_export_manifest()
            self.is_running = False