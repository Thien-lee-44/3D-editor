import os
import math
import json
import copy
import shutil
import numpy as np
import glm
from typing import Dict, List, Any, Optional, Tuple

from src.engine.resources.resource_manager import ResourceManager
from src.engine.scene.entity import Entity
from src.engine.scene.components import TransformComponent, MeshRenderer, LightComponent, CameraComponent

from src.app.exceptions import SimulationError, ResourceError
from src.app.config import MAX_LIGHTS, PASTE_OFFSET, MTL_TOKENS

class SceneManager:
    """
    High-level central controller dictating complex scene operations. 
    Handles logical data routing, hierarchy synchronization, deep serialization 
    (Save/Load/Snapshots), and exporting geometry formats.
    """
    
    def __init__(self, scene: Any) -> None:
        self.scene = scene
        self.clipboard: Optional[Entity] = None

    # =========================================================================
    # GENERIC ENTITY API DELEGATION
    # =========================================================================
    
    def has_clipboard(self) -> bool: 
        return self.clipboard is not None
        
    def get_selected_entity_id(self) -> int: 
        return self.scene.selected_index
        
    def select_entity(self, idx: int) -> None: 
        self.scene.selected_index = idx
        
    def set_manipulation_mode(self, mode: str) -> None: 
        self.scene.manipulation_mode = mode
        
    def get_scene_entities_list(self) -> List[Dict[str, Any]]:
        """Translates the multi-dimensional ECS tree into a flattened hierarchical dictionary list."""
        return [
            {
                "id": i, 
                "name": e.name, 
                "parent": None if not e.parent else self.scene.entities.index(e.parent), 
                "is_group": e.is_group
            } for i, e in enumerate(self.scene.entities)
        ]

    def reset_entity_transform(self, index: int) -> None:
        """Restores the local identity matrix coordinates of a specified entity."""
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
        """Translates spherical GUI coordinates into an underlying Quaternion rotation."""
        if self.scene.selected_index < 0: 
            return
            
        tf = self.scene.entities[self.scene.selected_index].get_component(TransformComponent)
        if tf:
            # Hướng mong muốn trong không gian thế giới
            world_quat = glm.angleAxis(glm.radians(yaw), glm.vec3(0, 1, 0)) * \
                         glm.angleAxis(glm.radians(pitch), glm.vec3(1, 0, 0))
            
            # Nếu có Parent, tính toán Rotation cục bộ: Local = Inverse(ParentWorld) * World
            if tf.entity and tf.entity.parent:
                parent_tf = tf.entity.parent.get_component(TransformComponent)
                if parent_tf:
                    parent_quat = parent_tf.global_quat_rot
                    tf.quat_rot = glm.inverse(parent_quat) * world_quat
                else:
                    tf.quat_rot = world_quat
            else:
                tf.quat_rot = world_quat
                
            tf.rotation = glm.degrees(glm.eulerAngles(tf.quat_rot))
            tf.sync_from_gui()

    def set_active_camera_selected(self) -> None:
        """Re-routes the primary rendering viewpoint to the currently selected camera component."""
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
        """Fetches the state of the active transform component based on current manipulation mode."""
        if self.scene.selected_index < 0: 
            return None
            
        tf = self.scene.entities[self.scene.selected_index].get_component(TransformComponent)
        if not tf: 
            return None
        
        mode = getattr(self.scene, 'manipulation_mode', 'ROTATE')
        val = tf.rotation if mode == "ROTATE" else (tf.position if mode == "MOVE" else tf.scale)
        
        return (mode, (val.x, val.y, val.z))

    def clear_scene(self) -> None:
        """Purges all entities and flushes the project asset cache."""
        self.scene.entities.clear()
        self.scene.selected_index = -1
        ResourceManager.clear_project_assets()
        self.scene.setup_default_camera()
        self.scene.setup_default_light()

    def get_selected_entity_data(self) -> Optional[Dict[str, Any]]:
        """Consolidates active component configurations into a unified data payload for the UI Inspector."""
        idx = self.scene.selected_index
        if idx < 0 or idx >= len(self.scene.entities): 
            return None
        
        ent = self.scene.entities[idx]
        data = {
            "index": idx, 
            "name": ent.name, 
            "is_group": ent.is_group, 
            "tf": None, 
            "mesh": None, 
            "light": None, 
            "cam": None
        }
        
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
        """Dynamically injects external UI input values directly back into the core ECS component states."""
        if self.scene.selected_index < 0: 
            return
            
        ent = self.scene.entities[self.scene.selected_index]
        if comp_name == "Entity": 
            setattr(ent, prop, value)
            return
        
        comp_class = {
            "Transform": TransformComponent, 
            "Mesh": MeshRenderer, 
            "Light": LightComponent, 
            "Camera": CameraComponent
        }.get(comp_name)
        
        if not comp_class: 
            return
            
        comp = ent.get_component(comp_class)
        if comp:
            if comp_name == "Transform" and prop in ["position", "rotation", "scale"]:
                setattr(comp, prop, glm.vec3(*value))
                if prop == "rotation": 
                    comp.quat_rot = glm.quat(glm.radians(comp.rotation))
                comp.sync_from_gui()
            elif prop.startswith("mat_"): 
                setattr(comp.material, prop[4:], glm.vec3(*value) if isinstance(value, list) else value)
            else: 
                setattr(comp, prop, glm.vec3(*value) if isinstance(value, list) and len(value)==3 else value)

    def copy_selected(self) -> None:
        """Records a deep copy of the selected entity state into the conceptual clipboard."""
        if self.scene.selected_index >= 0: 
            self.clipboard = copy.deepcopy(self.scene.entities[self.scene.selected_index])
    
    def cut_selected(self) -> None:
        """Moves the selected entity state to the clipboard and purges it from the active scene."""
        self.copy_selected()
        self.delete_selected()
        
    def _add_entity_recursive(self, ent: Entity) -> None:
        """Traverses and registers a hierarchical tree of entities into the main scene registry."""
        self.scene.add_entity(ent)
        for child in ent.children:
            self._add_entity_recursive(child)

    def paste_copied(self) -> None:
        """Clones the clipboard entity into the scene, validating hardware lighting limits."""
        if getattr(self, 'clipboard', None):
            light = self.clipboard.get_component(LightComponent)
            if light:
                ltype = light.type
                count = sum(1 for e in self.scene.entities if e.get_component(LightComponent) and e.get_component(LightComponent).type == ltype)
                
                limit = MAX_LIGHTS.get(ltype, 0)
                if count >= limit: 
                    raise SimulationError(f"Cannot paste {ltype} Light. Limit of {limit} reached.")
            
            new_ent = copy.deepcopy(self.clipboard)
            new_ent.name += " (Copy)"
            
            tf = new_ent.get_component(TransformComponent)
            if tf: 
                tf.position += glm.vec3(*PASTE_OFFSET) 
            
            self._add_entity_recursive(new_ent)
            return
            
        raise SimulationError("Cannot paste: The clipboard is currently empty.")
        
    def delete_selected(self) -> None:
        """Removes the actively selected entity and its children from the scene."""
        if self.scene.selected_index >= 0: 
            self.scene.remove_entity(self.scene.selected_index)

    def toggle_visibility_selected(self) -> None:
        """Toggles the renderable state of the selected entity."""
        if self.scene.selected_index >= 0:
            ent = self.scene.entities[self.scene.selected_index]
            light = ent.get_component(LightComponent)
            
            if light and light.type == "Directional": 
                return 
            
            mesh = ent.get_component(MeshRenderer)
            if mesh: 
                mesh.visible = not mesh.visible

    def toggle_all_lights(self, is_on: bool) -> None:
        """Globally toggles the illumination states of all light sources."""
        for ent in self.scene.entities:
            light = ent.get_component(LightComponent)
            if light: 
                light.on = is_on

    def toggle_all_proxies(self, is_visible: bool) -> None:
        """Globally toggles the visibility of Editor-only proxy meshes."""
        for ent in self.scene.entities:
            mesh = ent.get_component(MeshRenderer)
            if mesh and getattr(mesh, 'is_proxy', False): 
                mesh.visible = is_visible

    def sync_hierarchy_from_ui(self, hierarchy_mapping: Dict[int, Optional[int]]) -> None:
        """
        Translates user drag & drop actions within the GUI into a structural remapping of the ECS framework.
        Uses a differential update approach to prevent matrix corruption (NaN/Infinity) 
        and safely preserve absolute world transforms.
        """
        for child_id, parent_id in hierarchy_mapping.items():
            if child_id >= len(self.scene.entities):
                continue
                
            child = self.scene.entities[child_id]
            current_parent = child.parent
            
            new_parent = self.scene.entities[parent_id] if parent_id is not None else None
            
            if current_parent != new_parent:
                if new_parent is not None and not new_parent.is_group:
                    continue

                if current_parent is not None:
                    current_parent.remove_child(child, keep_world=True)
                    
                if new_parent is not None:
                    new_parent.add_child(child, keep_world=True)

    def load_texture_to_selected(self, map_attr: str, filepath: str) -> None:
        """Submits an I/O request to load an image map to a selected material channel."""
        if self.scene.selected_index < 0: 
            raise SimulationError("Please select an entity in the scene first!")
            
        mesh = self.scene.entities[self.scene.selected_index].get_component(MeshRenderer)
        
        if mesh:
            tex_id = ResourceManager.load_texture(filepath)
            setattr(mesh.material, map_attr, tex_id)
            
            if not hasattr(mesh.material, 'tex_paths'):
                mesh.material.tex_paths = {}
            mesh.material.tex_paths[map_attr] = filepath

    def remove_texture_from_selected(self, map_attr: str) -> None:
        """Removes the texture mapping from the specified material channel."""
        if self.scene.selected_index < 0: 
            return
            
        mesh = self.scene.entities[self.scene.selected_index].get_component(MeshRenderer)
        if mesh and getattr(mesh.material, map_attr, 0) != 0:
            setattr(mesh.material, map_attr, 0)
            
            if hasattr(mesh.material, 'tex_paths') and map_attr in mesh.material.tex_paths:
                del mesh.material.tex_paths[map_attr]

    def is_texture_in_use(self, path: str) -> bool:
        """Walks the entire scene graph to verify whether an asset dependency is actively leveraged."""
        def check_usage(ent: Entity) -> bool:
            mesh = ent.get_component(MeshRenderer)
            if mesh:
                if hasattr(mesh.material, 'tex_paths') and path in mesh.material.tex_paths.values():
                    return True
                    
            for child in ent.children:
                if check_usage(child): 
                    return True
            return False
            
        for ent in self.scene.entities:
            if check_usage(ent): 
                return True
                
        return False

    # =========================================================================
    # SERIALIZATION & EXPORT
    # =========================================================================

    def get_scene_snapshot(self) -> str:
        """Generates a JSON snapshot of the current scene state for the Undo stack."""
        entities_data = []
        for ent in self.scene.entities:
            if ent.parent is None:
                entities_data.append(self._serialize_entity(ent))
                
        return json.dumps(entities_data)

    def restore_snapshot(self, snapshot_str: str, current_aspect: float) -> None:
        """Reconstructs the scene state from a JSON snapshot payload."""
        if not snapshot_str: 
            return
            
        entities_data = json.loads(snapshot_str)
        
        self.scene.entities.clear()
        self.scene.selected_index = -1
        
        for ent_data in entities_data:
            ent = self._deserialize_entity(ent_data)
            self._add_entity_recursive(ent)

    def save_project(self, file_path: str, metadata: Dict[str, Any]) -> None:
        """Serializes the comprehensive project workspace into a discrete JSON file."""
        data = {
            "metadata": metadata,
            "assets": {
                "models": list(ResourceManager.project_models),
                "textures": list(ResourceManager.project_textures)
            },
            "entities": []
        }
        
        for ent in self.scene.entities:
            if ent.parent is None:
                data["entities"].append(self._serialize_entity(ent))

        with open(file_path, 'w', encoding='utf-8') as f: 
            json.dump(data, f, indent=4, ensure_ascii=False)

    def load_project(self, file_path: str, current_aspect: float) -> Dict[str, Any]:
        """Deserializes a JSON project file, restoring assets and rebuilding the entity tree."""
        with open(file_path, 'r', encoding='utf-8') as f: 
            data = json.load(f)

        self.scene.entities.clear()
        self.scene.selected_index = -1
        ResourceManager.clear_project_assets()
        
        for p in data.get("assets", {}).get("models", []): 
            ResourceManager.add_project_model(p)
            
        for p in data.get("assets", {}).get("textures", []): 
            ResourceManager.add_project_texture(p)
        
        for ent_data in data.get("entities", []):
            ent = self._deserialize_entity(ent_data)
            self._add_entity_recursive(ent)

        return data.get("metadata", {})

    def _serialize_entity(self, ent: Entity) -> Dict[str, Any]:
        """Deeply serializes a single entity and its children recursively."""
        data = {"name": ent.name, "is_group": ent.is_group, "components": {}, "children": []}
        
        tf = ent.get_component(TransformComponent)
        if tf: 
            data["components"]["transform"] = tf.to_dict()
            
        mesh = ent.get_component(MeshRenderer)
        if mesh: 
            data["components"]["mesh"] = mesh.to_dict()
            
        light = ent.get_component(LightComponent)
        if light: 
            data["components"]["light"] = light.to_dict()
            
        cam = ent.get_component(CameraComponent)
        if cam: 
            data["components"]["camera"] = cam.to_dict()
            
        for child in ent.children:
            data["children"].append(self._serialize_entity(child))
            
        return data

    def _deserialize_entity(self, data: Dict[str, Any], parent: Optional[Entity] = None) -> Entity:
        """Deeply reconstructs an entity and its components from a dictionary payload."""
        ent = Entity(data["name"], is_group=data.get("is_group", False))
        comps = data.get("components", {})
        
        if "transform" in comps:
            tf = ent.add_component(TransformComponent())
            tf.from_dict(comps["transform"])
            
        if "mesh" in comps:
            renderer = ent.add_component(MeshRenderer())
            renderer.from_dict(comps["mesh"])
            
        if "light" in comps:
            l_comp = comps["light"]
            light = ent.add_component(LightComponent(light_type=l_comp.get("type", "Point")))
            light.from_dict(l_comp)

        if "camera" in comps:
            c_comp = comps["camera"]
            cam = ent.add_component(CameraComponent(mode=c_comp.get("mode", "Perspective")))
            cam.from_dict(c_comp)
            
        for child_data in data.get("children", []):
            child_ent = self._deserialize_entity(child_data, ent) 
            ent.add_child(child_ent, keep_world=False)
            
        return ent

    def export_scene_obj(self, export_dir: str) -> None:
        """Exports all valid meshes in the scene to a Wavefront OBJ and MTL file format."""
        obj_path = os.path.join(export_dir, "models.obj")
        mtl_path = os.path.join(export_dir, "materials.mtl")
        tex_folder_name = "textures" 
        tex_dir_full = os.path.join(export_dir, tex_folder_name)

        with open(obj_path, 'w', encoding='utf-8') as f_obj, open(mtl_path, 'w', encoding='utf-8') as f_mtl:
            f_obj.write("mtllib materials.mtl\n")
            
            v_offset = 1 
            mat_count = 0

            def export_node(ent: Entity) -> None:
                nonlocal v_offset, mat_count
                mesh = ent.get_component(MeshRenderer)
                tf = ent.get_component(TransformComponent)

                if mesh and mesh.visible and not getattr(mesh, 'is_proxy', False) and mesh.geometry:
                    
                    mat_name = f"mat_{mat_count}_{ent.name.replace(' ', '_')}"
                    mat_count += 1
                    mat = mesh.material
                    
                    f_mtl.write(f"newmtl {mat_name}\n")
                    f_mtl.write(f"Ka {mat._ambient.x:.4f} {mat._ambient.y:.4f} {mat._ambient.z:.4f}\n")
                    f_mtl.write(f"Kd {mat._diffuse.x:.4f} {mat._diffuse.y:.4f} {mat._diffuse.z:.4f}\n")
                    f_mtl.write(f"Ks {mat._specular.x:.4f} {mat._specular.y:.4f} {mat._specular.z:.4f}\n")
                    f_mtl.write(f"Ns {getattr(mat, 'shininess', 32.0):.4f}\n")

                    # Iterates strictly through the modern tex_paths dictionary
                    tex_paths_dict = getattr(mat, 'tex_paths', {})

                    for map_attr, t_path in tex_paths_dict.items():
                        if t_path and os.path.exists(t_path):
                            tex_name = os.path.basename(t_path)
                            if not os.path.exists(tex_dir_full): 
                                os.makedirs(tex_dir_full)
                                
                            dest_tex = os.path.join(tex_dir_full, tex_name)
                            try:
                                if not os.path.exists(dest_tex) or not os.path.samefile(t_path, dest_tex):
                                    shutil.copy2(t_path, dest_tex)
                                
                                token = MTL_TOKENS.get(map_attr, "map_Kd")
                                f_mtl.write(f"{token} {tex_folder_name}/{tex_name}\n")
                            except Exception as e:
                                raise ResourceError(f"Texture packaging failed during export: {e}")
                                
                    f_mtl.write("\n")

                    f_obj.write(f"o {ent.name.replace(' ', '_')}\n")
                    f_obj.write(f"usemtl {mat_name}\n")

                    geom = mesh.geometry
                    if not hasattr(geom, 'vertices') or not hasattr(geom, 'vertex_size'): 
                        return
                    
                    v_size = geom.vertex_size
                    verts = np.array(geom.vertices, dtype=np.float32).reshape(-1, v_size)
                    num_v = len(verts)

                    global_mat = tf.get_matrix() if tf else glm.mat4(1.0)
                    norm_mat = glm.transpose(glm.inverse(glm.mat3(global_mat)))

                    for i in range(num_v):
                        pos = glm.vec3(verts[i][0], verts[i][1], verts[i][2])
                        g_pos = glm.vec3(global_mat * glm.vec4(pos, 1.0))
                        f_obj.write(f"v {g_pos.x:.6f} {g_pos.y:.6f} {g_pos.z:.6f}\n")

                    for i in range(num_v):
                        f_obj.write(f"vt {verts[i][6]:.6f} {verts[i][7]:.6f}\n")

                    for i in range(num_v):
                        norm = glm.vec3(verts[i][3], verts[i][4], verts[i][5])
                        g_norm = glm.normalize(norm_mat * norm) if glm.length(norm) > 0 else glm.vec3(0, 1, 0)
                        f_obj.write(f"vn {g_norm.x:.6f} {g_norm.y:.6f} {g_norm.z:.6f}\n")

                    if getattr(geom, 'indices', None) is not None and len(geom.indices) > 0:
                        idx = geom.indices
                        for i in range(0, len(idx), 3):
                            i1, i2, i3 = idx[i], idx[i+1], idx[i+2]
                            f_obj.write(f"f {i1+v_offset}/{i1+v_offset}/{i1+v_offset} "
                                        f"{i2+v_offset}/{i2+v_offset}/{i2+v_offset} "
                                        f"{i3+v_offset}/{i3+v_offset}/{i3+v_offset}\n")
                    else:
                        for i in range(num_v):
                            f_obj.write(f"p {i+v_offset}\n")
                            
                    v_offset += num_v

                for child in ent.children:
                    export_node(child)

            try:
                for ent in self.scene.entities:
                    if ent.parent is None:
                        export_node(ent)
            except Exception as e:
                raise ResourceError(f"Failed to export scene geometry to '{export_dir}'.\nReason: {e}")