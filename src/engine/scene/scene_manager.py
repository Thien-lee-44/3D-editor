"""
Scene Manager.

High-level central controller dictating complex scene operations. 
Strictly follows the Facade and Single Responsibility Principles.
Acts exclusively as a Router to specialized Sub-Managers.
"""

import os
import glm
import math
from typing import Dict, List, Any, Optional, Tuple

from src.engine.resources.resource_manager import ResourceManager
from src.engine.scene.entity import Entity
from src.engine.scene.components import TransformComponent, MeshRenderer, LightComponent, CameraComponent
from src.engine.scene.components.animation_cmp import AnimationComponent
from src.engine.scene.components.semantic_cmp import SemanticComponent

from src.engine.scene.managers.serialization_manager import SerializationManager
from src.engine.scene.managers.hierarchy_manager import HierarchyManager
from src.engine.scene.managers.clipboard_manager import ClipboardManager
from src.engine.scene.managers.animation_manager import AnimationBackendManager
from src.engine.scene.managers.semantic_manager import SemanticManager


class SceneManager:
    """
    Facade exposing unified commands for the UI layer to mutate the Scene Graph.
    Delegates internal logic to Clipboard, Hierarchy, Semantic, and Serialization sub-managers.
    """
    
    _COMPONENT_MAP = {
        "Transform": TransformComponent, 
        "Mesh": MeshRenderer, 
        "Light": LightComponent, 
        "Camera": CameraComponent,
        "Animation": AnimationComponent,
        "Semantic": SemanticComponent
    }

    def __init__(self, scene: Any) -> None:
        self.scene = scene
        self.serializer = SerializationManager(self.scene, self)
        self.hierarchy = HierarchyManager(self.scene, self)
        self.clipboard = ClipboardManager(self.scene, self)
        self.animation = AnimationBackendManager(self.scene)
        self.semantic = SemanticManager(self.scene)

    def _add_entity_recursive(self, ent: Entity) -> None:
        """Recursively registers an entity and its hierarchy into the scene."""
        self.scene.add_entity(ent)
        for child in ent.children:
            self._add_entity_recursive(child)

    # =========================================================================
    # CORE ENTITY MANAGEMENT
    # =========================================================================
    
    def get_selected_entity_id(self) -> int: return self.scene.selected_index
    def select_entity(self, idx: int) -> None: self.scene.selected_index = idx
    def set_manipulation_mode(self, mode: str) -> None: self.scene.manipulation_mode = mode
        
    def get_scene_entities_list(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": i, 
                "name": e.name, 
                "parent": None if not e.parent else self.scene.entities.index(e.parent), 
                "is_group": e.is_group
            } for i, e in enumerate(self.scene.entities)
        ]

    def reset_entity_transform(self, index: int) -> None:
        if index < 0 or index >= len(self.scene.entities): return
        tf = self.scene.entities[index].get_component(TransformComponent)
        if tf:
            if not tf.locked_axes.get("pos", False): tf.position = glm.vec3(0.0)
            if not tf.locked_axes.get("rot", False): 
                tf.rotation = glm.vec3(0.0)
                tf.quat_rot = glm.quat(glm.vec3(0.0))
            if not tf.locked_axes.get("scl", False): tf.scale = glm.vec3(1.0)
                
            tf.is_dirty = True
            if hasattr(tf, 'sync_from_gui'): tf.sync_from_gui()

    def update_light_direction(self, yaw: float, pitch: float) -> None:
        if self.scene.selected_index < 0: return
        tf = self.scene.entities[self.scene.selected_index].get_component(TransformComponent)
        if tf and not tf.locked_axes.get("rot", False):
            world_quat = glm.angleAxis(glm.radians(yaw), glm.vec3(0, 1, 0)) * \
                         glm.angleAxis(glm.radians(pitch), glm.vec3(1, 0, 0))
            if tf.entity and tf.entity.parent:
                parent_tf = tf.entity.parent.get_component(TransformComponent)
                tf.quat_rot = glm.inverse(parent_tf.global_quat_rot) * world_quat if parent_tf else world_quat
            else:
                tf.quat_rot = world_quat
                
            tf.rotation = glm.degrees(glm.eulerAngles(tf.quat_rot))
            tf.is_dirty = True
            if hasattr(tf, 'sync_from_gui'): tf.sync_from_gui()

    def set_active_camera_selected(self) -> None:
        if self.scene.selected_index < 0: return
        ent = self.scene.entities[self.scene.selected_index]
        target_cam = ent.get_component(CameraComponent)
        if target_cam:
            for e in self.scene.entities:
                c = e.get_component(CameraComponent)
                if c: 
                    c.is_active = False
                    m = e.get_component(MeshRenderer)
                    if m: m.visible = True
            target_cam.is_active = True
            m_target = ent.get_component(MeshRenderer)
            if m_target: m_target.visible = False

    def get_selected_transform_state(self) -> Optional[Tuple[str, Tuple[float, float, float]]]:
        if self.scene.selected_index < 0: return None
        tf = self.scene.entities[self.scene.selected_index].get_component(TransformComponent)
        if not tf: return None
        mode = getattr(self.scene, 'manipulation_mode', 'ROTATE')
        if mode == "NONE": return None
        val = tf.rotation if mode == "ROTATE" else (tf.position if mode == "MOVE" else tf.scale)
        return (mode, (val.x, val.y, val.z))

    def clear_scene(self) -> None:
        self.scene.clear_entities()
        ResourceManager.clear_project_assets()

    def get_selected_entity_data(self) -> Optional[Dict[str, Any]]:
        idx = self.scene.selected_index
        if idx < 0 or idx >= len(self.scene.entities): return None
        ent = self.scene.entities[idx]
        
        data = {
            "index": idx, "name": ent.name, "is_group": ent.is_group, 
            "tf": None, "mesh": None, "light": None, "cam": None, "anim": None, "semantic": None
        }
        
        comp_keys = {
            TransformComponent: "tf", MeshRenderer: "mesh", LightComponent: "light",
            CameraComponent: "cam", AnimationComponent: "anim", SemanticComponent: "semantic"
        }
        
        for comp_cls, dict_key in comp_keys.items():
            comp = ent.get_component(comp_cls)
            if comp:
                if hasattr(comp, 'to_dict'): data[dict_key] = comp.to_dict()
                elif hasattr(comp, 'serialize'): data[dict_key] = comp.serialize()

        if data["light"] and data["tf"] and data["light"].get("type") in ["Directional", "Spot"]:
            tf = ent.get_component(TransformComponent)
            forward = glm.vec3(glm.mat4_cast(tf.global_quat_rot) * glm.vec4(0, 0, -1, 0))
            data["light"]["pitch"] = math.degrees(math.asin(max(-1.0, min(1.0, forward.y))))
            y_val = math.degrees(math.atan2(-forward.x, -forward.z))
            data["light"]["yaw"] = y_val if y_val >= 0 else y_val + 360.0
            
        return data

    def set_component_property(self, comp_name: str, prop: str, value: Any) -> None:
        """Legacy singular property update method. Maintained for backwards compatibility."""
        self.set_component_properties(comp_name, {prop: value})

    def set_component_properties(self, comp_name: str, payload: Dict[str, Any]) -> None:
        """Atomic payload execution for immediate entity state synchronization."""
        if self.scene.selected_index < 0: return
        ent = self.scene.entities[self.scene.selected_index]
        
        if comp_name == "Entity": 
            for prop, value in payload.items():
                setattr(ent, prop, value)
            return
            
        comp_class = self._COMPONENT_MAP.get(comp_name)
        if not comp_class: return
        comp = ent.get_component(comp_class)
        if not comp: return

        if comp_name == "Animation":
            for prop, value in payload.items():
                self.animation.handle_animation_property(ent, comp, prop, value)
            return
            
        if comp_name == "Semantic":
            for prop, value in payload.items():
                self.semantic.handle_semantic_property(ent, comp, prop, value)
            return

        for prop, value in payload.items():
            if comp_name == "Transform" and prop in ["position", "rotation", "scale"]:
                if prop == "position" and comp.locked_axes.get("pos", False): continue
                if prop == "rotation" and comp.locked_axes.get("rot", False): continue
                if prop == "scale" and comp.locked_axes.get("scl", False): continue                
                setattr(comp, prop, glm.vec3(*value))
                if prop == "rotation": 
                    comp.quat_rot = glm.quat(glm.radians(comp.rotation))
                comp.is_dirty = True
                if hasattr(comp, 'sync_from_gui'):
                    comp.sync_from_gui()
            elif prop.startswith("mat_"): 
                setattr(comp.material, prop[4:], glm.vec3(*value) if isinstance(value, list) else value)
            else: 
                setattr(comp, prop, glm.vec3(*value) if isinstance(value, list) and len(value) == 3 else value)

    def toggle_visibility_selected(self) -> None:
        if self.scene.selected_index >= 0:
            ent = self.scene.entities[self.scene.selected_index]
            light = ent.get_component(LightComponent)
            if light and light.type == "Directional": return 
            mesh = ent.get_component(MeshRenderer)
            if mesh: mesh.visible = not mesh.visible

    def toggle_all_lights(self, is_on: bool) -> None:
        for ent in self.scene.entities:
            light = ent.get_component(LightComponent)
            if light: light.on = is_on

    def toggle_all_proxies(self, is_visible: bool) -> None:
        for ent in self.scene.entities:
            mesh = ent.get_component(MeshRenderer)
            if mesh and getattr(mesh, 'is_proxy', False): mesh.visible = is_visible

    # =========================================================================
    # RESOURCE & FACADE ROUTING
    # =========================================================================

    def load_texture_to_selected(self, map_attr: str, filepath: str) -> None:
        if self.scene.selected_index < 0: return
        mesh = self.scene.entities[self.scene.selected_index].get_component(MeshRenderer)
        if mesh:
            normalized_path = os.path.normpath(os.path.abspath(filepath))
            tex_id = ResourceManager.load_texture(normalized_path)
            if tex_id == 0:
                return

            if hasattr(mesh.material, "get_tex_paths_snapshot") and hasattr(mesh.material, "apply_texture_paths"):
                new_paths = mesh.material.get_tex_paths_snapshot()
                new_paths[map_attr] = normalized_path
                mesh.material.apply_texture_paths(new_paths)
            else:
                setattr(mesh.material, map_attr, tex_id)
                if not hasattr(mesh.material, 'tex_paths'):
                    mesh.material.tex_paths = {}
                mesh.material.tex_paths[map_attr] = normalized_path

    def remove_texture_from_selected(self, map_attr: str) -> None:
        if self.scene.selected_index < 0: return
        mesh = self.scene.entities[self.scene.selected_index].get_component(MeshRenderer)
        if mesh and getattr(mesh.material, map_attr, 0) != 0:
            if hasattr(mesh.material, "get_tex_paths_snapshot") and hasattr(mesh.material, "apply_texture_paths"):
                new_paths = mesh.material.get_tex_paths_snapshot()
                if map_attr in new_paths:
                    del new_paths[map_attr]
                mesh.material.apply_texture_paths(new_paths)
            else:
                setattr(mesh.material, map_attr, 0)
                if hasattr(mesh.material, 'tex_paths') and map_attr in mesh.material.tex_paths:
                    del mesh.material.tex_paths[map_attr]

    def is_texture_in_use(self, path: str) -> bool:
        target_path = os.path.normcase(os.path.normpath(os.path.abspath(path)))
        for ent in self.scene.entities:
            if self._check_texture_usage(ent, target_path): return True
        return False

    def _check_texture_usage(self, ent: Entity, path: str) -> bool:
        mesh = ent.get_component(MeshRenderer)
        if mesh:
            if hasattr(mesh.material, "get_tex_paths_snapshot"):
                tex_values = mesh.material.get_tex_paths_snapshot().values()
            else:
                tex_values = getattr(mesh.material, "tex_paths", {}).values()

            normalized_values = {
                os.path.normcase(os.path.normpath(os.path.abspath(v)))
                for v in tex_values
                if isinstance(v, str) and v.strip()
            }
            if path in normalized_values:
                return True
        for child in ent.children:
            if self._check_texture_usage(child, path): return True
        return False

    # Facade Delegations
    def get_scene_snapshot(self) -> str: return self.serializer.get_scene_snapshot()
    def restore_snapshot(self, snapshot_str: str, current_aspect: float) -> None: self.serializer.restore_snapshot(snapshot_str, current_aspect)
    def save_project(self, file_path: str, metadata: Dict[str, Any]) -> None: self.serializer.save_project(file_path, metadata)
    def load_project(self, file_path: str, current_aspect: float) -> Dict[str, Any]: return self.serializer.load_project(file_path, current_aspect)
    def export_scene_obj(self, export_dir: str) -> None: self.serializer.export_scene_obj(export_dir)

    def group_selected_entities(self, entity_ids: List[int]) -> None: self.hierarchy.group_selected_entities(entity_ids)
    def ungroup_selected_entity(self) -> None: self.hierarchy.ungroup_selected_entity()
    def sync_hierarchy_from_ui(self, hierarchy_mapping: Dict[int, Optional[int]]) -> None: self.hierarchy.sync_hierarchy_from_ui(hierarchy_mapping)

    def has_clipboard(self) -> bool: return self.clipboard.has_clipboard()
    def copy_selected(self) -> None: self.clipboard.copy_selected()
    def cut_selected(self) -> None: self.clipboard.cut_selected()
    def paste_copied(self) -> None: self.clipboard.paste_copied()
    def delete_selected(self) -> None: self.clipboard.delete_selected()

    def get_semantic_classes(self) -> dict: return self.semantic.get_semantic_classes()
    def add_semantic_class(self, name: str) -> int: return self.semantic.add_semantic_class(name)
    def update_semantic_class_color(self, class_id: int, color: list) -> None: self.semantic.update_semantic_class_color(class_id, color)
    def remove_semantic_class(self, class_id: int) -> None: self.semantic.remove_semantic_class(class_id)
    
    def get_animation_info(self) -> dict: return self.animation.get_animation_info()
    def set_active_keyframe(self, index: int) -> float: return self.animation.set_active_keyframe(index)
    def sync_gizmo_to_keyframe(self, current_time: float) -> bool: return self.animation.sync_gizmo_to_keyframe(current_time)
    
    def update_keyframe_property(self, current_time: float, comp_name: str, prop: str, value: Any) -> tuple: 
        return self.animation.update_keyframe_property(current_time, comp_name, prop, value)
        
    def update_keyframe_properties(self, current_time: float, comp_name: str, payload: Dict[str, Any]) -> tuple: 
        return self.animation.update_keyframe_properties(current_time, comp_name, payload)
        
    def add_and_focus_keyframe(self, time: float) -> int: return self.animation.add_and_focus_keyframe(time)
    
    def handle_animation_property(self, ent: Entity, comp: AnimationComponent, prop: str, value: Any) -> None:
        self.animation.handle_animation_property(ent, comp, prop, value)