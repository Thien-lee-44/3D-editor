"""
Scene Manager Facade.
Provides a unified API for interacting with the Scene, routing complex operations
to specialized sub-managers (Hierarchy, Serialization, Clipboard).
"""

import math
import glm
from typing import Dict, List, Any, Optional, Tuple

from src.engine.resources.resource_manager import ResourceManager
from src.engine.scene.entity import Entity
from src.engine.scene.components import TransformComponent, MeshRenderer, LightComponent, CameraComponent
from src.engine.scene.managers.serialization_manager import SerializationManager
from src.engine.scene.managers.hierarchy_manager import HierarchyManager
from src.engine.scene.managers.clipboard_manager import ClipboardManager
from src.app.exceptions import SimulationError

class SceneManager:
    """
    High-level facade orchestrating state modifications requested by the UI Controllers.
    """

    _COMPONENT_MAP = {
        "Transform": TransformComponent,
        "Mesh": MeshRenderer,
        "Light": LightComponent,
        "Camera": CameraComponent,
    }

    def __init__(self, scene: Any) -> None:
        self.scene = scene
        self.serializer = SerializationManager(self.scene, self)
        self.hierarchy = HierarchyManager(self.scene, self)
        self.clipboard = ClipboardManager(self.scene, self)

    def _add_entity_recursive(self, ent: Entity) -> None:
        """Recursively registers a cloned or loaded hierarchy into the Scene."""
        self.scene.add_entity(ent)
        for child in ent.children:
            self._add_entity_recursive(child)

    def get_selected_entity_id(self) -> int:
        return self.scene.selected_index

    def select_entity(self, idx: int) -> None:
        self.scene.selected_index = idx

    def set_manipulation_mode(self, mode: str) -> None:
        self.scene.manipulation_mode = mode

    def get_scene_entities_list(self) -> List[Dict[str, Any]]:
        """Extracts a flat list of entities for UI TreeWidget population."""
        return [
            {
                "id": i,
                "name": e.name,
                "parent": None if not e.parent else self.scene.entities.index(e.parent),
                "is_group": e.is_group,
            }
            for i, e in enumerate(self.scene.entities)
        ]

    def reset_entity_transform(self, index: int) -> None:
        """Resets position, rotation, and scale to identity."""
        if index < 0 or index >= len(self.scene.entities):
            return
        tf = self.scene.entities[index].get_component(TransformComponent)
        if tf:
            tf.position = glm.vec3(0.0)
            tf.rotation = glm.vec3(0.0)
            tf.scale = glm.vec3(1.0)
            tf.quat_rot = glm.quat(glm.vec3(0.0))
            tf.sync_from_gui()

    def update_light_direction(self, yaw: float, pitch: float) -> None:
        """Updates the rotation of directional or spot lights using spherical coordinates."""
        if self.scene.selected_index < 0:
            return
        tf = self.scene.entities[self.scene.selected_index].get_component(TransformComponent)
        if tf:
            world_quat = glm.angleAxis(glm.radians(yaw), glm.vec3(0, 1, 0)) * glm.angleAxis(glm.radians(pitch), glm.vec3(1, 0, 0))
            if tf.entity and tf.entity.parent:
                parent_tf = tf.entity.parent.get_component(TransformComponent)
                if parent_tf:
                    tf.quat_rot = glm.inverse(parent_tf.global_quat_rot) * world_quat
                else:
                    tf.quat_rot = world_quat
            else:
                tf.quat_rot = world_quat
                
            tf.rotation = glm.degrees(glm.eulerAngles(tf.quat_rot))
            tf.sync_from_gui()

    def set_active_camera_selected(self) -> None:
        """Swaps the rendering context to use the currently selected Camera entity."""
        if self.scene.selected_index < 0:
            return
        ent = self.scene.entities[self.scene.selected_index]
        target_cam = ent.get_component(CameraComponent)
        if target_cam:
            for e in self.scene.entities:
                c = e.get_component(CameraComponent)
                if c:
                    c.is_active = False
                    m = e.get_component(MeshRenderer)
                    if m:
                        m.visible = True
                        
            target_cam.is_active = True
            m_target = ent.get_component(MeshRenderer)
            if m_target:
                m_target.visible = False

    def get_selected_transform_state(self) -> Optional[Tuple[str, Tuple[float, float, float]]]:
        """Returns the specific transform axis values mapped to the active Gizmo mode."""
        if self.scene.selected_index < 0:
            return None
        tf = self.scene.entities[self.scene.selected_index].get_component(TransformComponent)
        if not tf:
            return None

        mode = getattr(self.scene, "manipulation_mode", "ROTATE")
        val = tf.rotation if mode == "ROTATE" else (tf.position if mode == "MOVE" else tf.scale)
        return (mode, (val.x, val.y, val.z))

    def clear_scene(self) -> None:
        """Flushes all entities and project manifests."""
        self.scene.clear_entities()
        ResourceManager.clear_project_assets()

    def get_selected_entity_data(self) -> Optional[Dict[str, Any]]:
        """Bundles the active state of all components attached to the selected entity for the Inspector UI."""
        idx = self.scene.selected_index
        if idx < 0 or idx >= len(self.scene.entities):
            return None

        ent = self.scene.entities[idx]
        data = {"index": idx, "name": ent.name, "is_group": ent.is_group, "tf": None, "mesh": None, "light": None, "cam": None}

        tf = ent.get_component(TransformComponent)
        if tf:
            data["tf"] = tf.to_dict()

        mesh = ent.get_component(MeshRenderer)
        if mesh:
            data["mesh"] = mesh.to_dict()

        light = ent.get_component(LightComponent)
        if light:
            data["light"] = light.to_dict()
            if tf and light.type in ["Directional", "Spot"]:
                forward = glm.vec3(glm.mat4_cast(tf.global_quat_rot) * glm.vec4(0, 0, -1, 0))
                data["light"]["pitch"] = math.degrees(math.asin(max(-1.0, min(1.0, forward.y))))
                y_val = math.degrees(math.atan2(-forward.x, -forward.z))
                data["light"]["yaw"] = y_val if y_val >= 0 else y_val + 360.0

        cam = ent.get_component(CameraComponent)
        if cam:
            data["cam"] = cam.to_dict()

        return data

    def set_component_property(self, comp_name: str, prop: str, value: Any) -> None:
        """Mutates a specific field inside a specific component."""
        if self.scene.selected_index < 0:
            return

        ent = self.scene.entities[self.scene.selected_index]
        if comp_name == "Entity":
            setattr(ent, prop, value)
            return

        comp_class = self._COMPONENT_MAP.get(comp_name)
        if not comp_class:
            return
        comp = ent.get_component(comp_class)
        if not comp:
            return

        if comp_name == "Transform" and prop in ["position", "rotation", "scale"]:
            setattr(comp, prop, glm.vec3(*value))
            if prop == "rotation":
                comp.quat_rot = glm.quat(glm.radians(comp.rotation))
            comp.sync_from_gui()
        elif prop.startswith("mat_"):
            setattr(comp.material, prop[4:], glm.vec3(*value) if isinstance(value, list) else value)
        else:
            setattr(comp, prop, glm.vec3(*value) if isinstance(value, list) and len(value) == 3 else value)

    def toggle_visibility_selected(self) -> None:
        """Toggles the visible state of the selected renderable."""
        if self.scene.selected_index >= 0:
            ent = self.scene.entities[self.scene.selected_index]
            light = ent.get_component(LightComponent)
            if light and light.type == "Directional":
                return
            mesh = ent.get_component(MeshRenderer)
            if mesh:
                mesh.visible = not mesh.visible

    def toggle_all_lights(self, is_on: bool) -> None:
        """Global override for lighting system execution."""
        for ent in self.scene.entities:
            light = ent.get_component(LightComponent)
            if light:
                light.on = is_on

    def toggle_all_proxies(self, is_visible: bool) -> None:
        """Global override for the visibility of editor utility meshes."""
        for ent in self.scene.entities:
            mesh = ent.get_component(MeshRenderer)
            if mesh and getattr(mesh, "is_proxy", False):
                mesh.visible = is_visible

    def load_texture_to_selected(self, map_attr: str, filepath: str) -> None:
        """Binds a texture to a specific material map slot on the selected entity."""
        if self.scene.selected_index < 0:
            raise SimulationError("Please select an entity in the scene first!")

        mesh = self.scene.entities[self.scene.selected_index].get_component(MeshRenderer)
        if mesh:
            tex_id = ResourceManager.load_texture(filepath)
            setattr(mesh.material, map_attr, tex_id)
            if not hasattr(mesh.material, "tex_paths"):
                mesh.material.tex_paths = {}
            mesh.material.tex_paths[map_attr] = filepath

    def remove_texture_from_selected(self, map_attr: str) -> None:
        """Unbinds a texture from a specific material map slot."""
        if self.scene.selected_index < 0:
            return

        mesh = self.scene.entities[self.scene.selected_index].get_component(MeshRenderer)
        if mesh and getattr(mesh.material, map_attr, 0) != 0:
            setattr(mesh.material, map_attr, 0)
            if hasattr(mesh.material, "tex_paths") and map_attr in mesh.material.tex_paths:
                del mesh.material.tex_paths[map_attr]

    def is_texture_in_use(self, path: str) -> bool:
        """Scans the active scene graph to check if a specific texture path is utilized."""
        for ent in self.scene.entities:
            if self._check_texture_usage(ent, path):
                return True
        return False

    def _check_texture_usage(self, ent: Entity, path: str) -> bool:
        mesh = ent.get_component(MeshRenderer)
        if mesh and hasattr(mesh.material, "tex_paths") and path in mesh.material.tex_paths.values():
            return True
        for child in ent.children:
            if self._check_texture_usage(child, path):
                return True
        return False

    # --- Sub-Manager Delegations ---
    def get_scene_snapshot(self) -> str:
        return self.serializer.get_scene_snapshot()

    def restore_snapshot(self, snapshot_str: str, current_aspect: float) -> None:
        self.serializer.restore_snapshot(snapshot_str, current_aspect)

    def save_project(self, file_path: str, metadata: Dict[str, Any]) -> None:
        self.serializer.save_project(file_path, metadata)

    def load_project(self, file_path: str, current_aspect: float) -> Dict[str, Any]:
        return self.serializer.load_project(file_path, current_aspect)

    def export_scene_obj(self, export_dir: str) -> None:
        self.serializer.export_scene_obj(export_dir)

    def group_selected_entities(self, entity_ids: List[int]) -> None:
        self.hierarchy.group_selected_entities(entity_ids)

    def ungroup_selected_entity(self) -> None:
        self.hierarchy.ungroup_selected_entity()

    def sync_hierarchy_from_ui(self, hierarchy_mapping: Dict[int, Optional[int]]) -> None:
        self.hierarchy.sync_hierarchy_from_ui(hierarchy_mapping)

    def has_clipboard(self) -> bool:
        return self.clipboard.has_clipboard()

    def copy_selected(self) -> None:
        self.clipboard.copy_selected()

    def cut_selected(self) -> None:
        self.clipboard.cut_selected()

    def paste_copied(self) -> None:
        self.clipboard.paste_copied()

    def delete_selected(self) -> None:
        self.clipboard.delete_selected()