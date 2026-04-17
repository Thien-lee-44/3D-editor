import os
import glm
from typing import Any, Optional, Dict, List
from src.engine.scene.entity import Entity
from src.engine.scene.components import TransformComponent, MeshRenderer, LightComponent, CameraComponent
from src.engine.geometry.primitives import PrimitivesManager
from src.engine.resources.resource_manager import ResourceManager
from src.engine.graphics.buffer_objects import BufferObject

from src.engine.scene.components.animation_cmp import AnimationComponent
from src.engine.scene.components.semantic_cmp import SemanticComponent
from src.engine.synthetic.tracking_mgr import TrackingManager

from src.app.exceptions import SimulationError, ResourceError
from src.app.config import (
    MAX_LIGHTS, DEFAULT_GROUP_NAME, DEFAULT_PROXY_SCALE,
    DEFAULT_CAMERA_NAME, DEFAULT_SCENE_CAM_POS, DEFAULT_SCENE_LIGHT_ROT
)


class EntityFactory:
    """
    Implements the Abstract Factory design pattern for assembling complex Entity configurations.
    Strictly adheres to the Single Responsibility Principle by decoupling Animation from Semantics.
    """
    
    def __init__(self, scene: Any) -> None:
        self.scene = scene

    def _attach_animation(self, ent: Entity) -> None:
        """Grants the entity the ability to be keyframed over time."""
        ent.add_component(AnimationComponent())

    def _attach_semantic(self, ent: Entity, class_id: int = 0, track_id: Optional[int] = None) -> int:
        """
        Grants the entity Ground Truth tracking data for AI dataset generation.
        Returns the assigned track_id to propagate to child entities (Semantic Inheritance).
        """
        if track_id is None:
            track_id = TrackingManager.get_next_id()
            
        ent.add_component(SemanticComponent(track_id=track_id, class_id=class_id))
        return track_id

    # =========================================================================
    # SCENE BOOTSTRAPPING
    # =========================================================================

    def setup_default_scene(self) -> None:
        """Bootstraps the mandatory viewing frustum, lighting, and default geometry."""
        
        # 1. Default Camera
        cam = Entity(DEFAULT_CAMERA_NAME)
        tf = cam.add_component(TransformComponent())
        tf.position = glm.vec3(*DEFAULT_SCENE_CAM_POS)
        tf.scale = glm.vec3(DEFAULT_PROXY_SCALE) 
        tf.locked_axes["scl"] = True  # [CONSTRAINT]
        
        self._attach_animation(cam)
        
        cam_comp = CameraComponent(mode="Perspective")
        cam_comp.is_active = True 
        cam.add_component(cam_comp)
        
        renderer = cam.add_component(MeshRenderer())
        renderer.is_proxy = True
        
        proxy_path = PrimitivesManager.get_proxy_path("proxy_camera.ply")
        if os.path.exists(proxy_path):
            mesh_list = ResourceManager.get_model(proxy_path)
            if mesh_list:
                sub = mesh_list[0]
                geom = BufferObject(sub.vertices, sub.indices, sub.vertex_size)
                geom.filepath = proxy_path
                renderer.geometry = geom
                
        self.scene.add_entity(cam)

        # 2. Default Light
        light = Entity("Directional Light")
        tf = light.add_component(TransformComponent())
        tf.rotation = glm.vec3(*DEFAULT_SCENE_LIGHT_ROT)
        tf.quat_rot = glm.quat(glm.radians(tf.rotation))
        tf.locked_axes["pos"] = True  # [CONSTRAINT]
        tf.locked_axes["scl"] = True  # [CONSTRAINT]
        
        self._attach_animation(light)
        light.add_component(LightComponent(light_type="Directional"))
        self.scene.add_entity(light)

        # 3. Default Cube
        cube_entity = Entity("Default Cube")
        cube_entity.add_component(TransformComponent())
        
        self._attach_animation(cube_entity)
        self._attach_semantic(cube_entity, class_id=3)
            
        renderer = cube_entity.add_component(MeshRenderer())
        geom = PrimitivesManager.get_primitive("Cube")
        if geom: 
            renderer.geometry = geom
            
        self.scene.add_entity(cube_entity)
        
    # =========================================================================
    # ENTITY SPAWNING
    # =========================================================================
    
    def add_empty_group(self) -> None:
        """Spawns an empty transform node primarily used for hierarchical grouping."""
        ent = Entity(DEFAULT_GROUP_NAME, is_group=True)
        tf = ent.add_component(TransformComponent())
        tf.locked_axes["scl"] = True  # [CONSTRAINT] Prevent shearing of child objects
        
        self._attach_animation(ent)
        self._attach_semantic(ent, class_id=3) 
        
        self.scene.add_entity(ent)

    def spawn_primitive(self, name: str, is_2d: bool) -> None:
        """Instantiates an entity equipped with a foundational geometric mesh."""
        geom = PrimitivesManager.get_primitive(name, is_2d)
        
        if geom:
            ent = Entity(name)
            ent.add_component(TransformComponent())
            
            self._attach_animation(ent)
            self._attach_semantic(ent, class_id=3) 
            
            renderer = ent.add_component(MeshRenderer())
            renderer.geometry = geom
            
            if hasattr(geom, 'materials') and 'default_active' in geom.materials:
                renderer.material.setup_from_dict(geom.materials['default_active'])
                
            self.scene.add_entity(ent)

    def spawn_math_surface(self, formula: str, xmin: float, xmax: float, ymin: float, ymax: float, res: int) -> None:
        """Instantiates an entity rendering a procedurally generated mathematical surface."""
        from src.engine.geometry.math_surface import MathSurface
        
        ent = Entity(f"Math: {formula}")
        ent.add_component(TransformComponent())
        
        self._attach_animation(ent)
        self._attach_semantic(ent, class_id=3) 
        
        renderer = ent.add_component(MeshRenderer())
        geom = MathSurface(formula, (xmin, xmax), (ymin, ymax), res)
        geom.formula_str = formula
        renderer.geometry = geom
        
        self.scene.add_entity(ent)

    def add_light(self, light_type: str, proxy_enabled: bool, global_light_on: bool) -> None:
        """Instantiates a light source. Lights are NEVER given a SemanticComponent."""
        current_count = sum(1 for _, l, _ in self.scene.cached_lights if l.type == light_type)
        limit = MAX_LIGHTS.get(light_type, 0)
        
        if current_count >= limit:
            raise SimulationError(f"Cannot add {light_type} Light. Maximum limit reached ({limit} lights).")

        ent = Entity(f"{light_type} Light")
        tf = ent.add_component(TransformComponent())
        
        # [CONSTRAINT RULES]
        if light_type == "Directional":
            tf.locked_axes["pos"] = True
            tf.locked_axes["scl"] = True
        elif light_type == "Point":
            tf.locked_axes["rot"] = True
            tf.locked_axes["scl"] = True
        elif light_type == "Spot":
            tf.locked_axes["scl"] = True
        
        self._attach_animation(ent) 
        
        light_comp = ent.add_component(LightComponent(light_type=light_type))
        light_comp.on = global_light_on
        
        if light_type != "Directional":
            renderer = ent.add_component(MeshRenderer())
            renderer.is_proxy = True
            renderer.visible = proxy_enabled
            
            if light_type == "Point": 
                renderer.geometry = PrimitivesManager.get_proxy("proxy_point.ply")
                tf.scale = glm.vec3(DEFAULT_PROXY_SCALE)
            elif light_type == "Spot": 
                renderer.geometry = PrimitivesManager.get_proxy("proxy_spot.ply")
                tf.scale = glm.vec3(DEFAULT_PROXY_SCALE)
                
        self.scene.add_entity(ent)

    def add_camera(self, proxy_enabled: bool) -> None:
        """Instantiates an auxiliary view frustum. Cameras are NEVER given a SemanticComponent."""
        ent = Entity("Camera")
        tf = ent.add_component(TransformComponent())
        tf.locked_axes["scl"] = True  # [CONSTRAINT]
        
        self._attach_animation(ent) 
        
        cam = ent.add_component(CameraComponent(mode="Perspective"))
        cam.is_active = not any(e.get_component(CameraComponent) for e in self.scene.entities)
        
        renderer = ent.add_component(MeshRenderer())
        renderer.is_proxy = True
        renderer.visible = proxy_enabled
        tf.scale = glm.vec3(DEFAULT_PROXY_SCALE)
        renderer.geometry = PrimitivesManager.get_proxy("proxy_camera.ply")
        
        self.scene.add_entity(ent)

    def spawn_model_from_path(self, path: str) -> None:
        """
        Parses a 3D asset and assembles a hierarchical structure.
        Automatically calculates the group center to align Gizmos with the collective children.
        """
        try:
            mesh_data_list = ResourceManager.get_model(path)
            raw_name = os.path.splitext(os.path.basename(path))[0]
            display_name = raw_name.replace('_', ' ').title()

            buckets: Dict[str, List[Any]] = {}
            all_child_positions = []

            for sub_data in mesh_data_list:
                g_name = getattr(sub_data, 'group_name', display_name)
                if g_name not in buckets:
                    buckets[g_name] = []
                buckets[g_name].append(sub_data)
                if hasattr(sub_data, 'pivot_offset'):
                    all_child_positions.append(glm.vec3(*sub_data.pivot_offset))

            master_center = glm.vec3(0.0)
            if all_child_positions:
                min_p = glm.vec3(min(p.x for p in all_child_positions), min(p.y for p in all_child_positions), min(p.z for p in all_child_positions))
                max_p = glm.vec3(max(p.x for p in all_child_positions), max(p.y for p in all_child_positions), max(p.z for p in all_child_positions))
                master_center = (min_p + max_p) / 2.0

            master_ent = Entity(display_name, is_group=True)
            master_tf = master_ent.add_component(TransformComponent())
            master_tf.position = master_center 
            master_tf.locked_axes["scl"] = True  # [CONSTRAINT] Prevent global hierarchy shearing
            
            self._attach_animation(master_ent)
            master_track_id = self._attach_semantic(master_ent, class_id=0)
            self.scene.add_entity(master_ent)

            for g_name, meshes in buckets.items():
                obj_world_positions = [glm.vec3(*m.pivot_offset) for m in meshes if hasattr(m, 'pivot_offset')]
                if not obj_world_positions: continue
                
                obj_world_center = sum(obj_world_positions, glm.vec3(0.0)) / len(obj_world_positions)
                
                is_multi_part = len(meshes) > 1
                obj_ent = Entity(g_name, is_group=is_multi_part)
                
                obj_tf = obj_ent.add_component(TransformComponent())
                obj_tf.position = obj_world_center - master_center
                if is_multi_part:
                    obj_tf.locked_axes["scl"] = True  # [CONSTRAINT]
                
                self._attach_animation(obj_ent)
                obj_track_id = self._attach_semantic(obj_ent, class_id=0, track_id=master_track_id)
                
                master_ent.add_child(obj_ent, keep_world=False)
                self.scene.add_entity(obj_ent)

                for sub_data in meshes:
                    if is_multi_part:
                        child_ent = Entity(getattr(sub_data, 'name', 'SubMesh'))
                        child_tf = child_ent.add_component(TransformComponent())
                        child_tf.position = glm.vec3(*sub_data.pivot_offset) - obj_world_center
                        
                        self._attach_animation(child_ent)
                        self._attach_semantic(child_ent, class_id=0, track_id=obj_track_id)
                        
                        renderer = child_ent.add_component(MeshRenderer())
                        renderer.geometry = BufferObject(sub_data.vertices, sub_data.indices, vertex_size=sub_data.vertex_size)
                        renderer.geometry.pivot_offset = sub_data.pivot_offset
                        
                        renderer.geometry.filepath = path
                        renderer.geometry.name = getattr(sub_data, 'name', '')
                        renderer.geometry.render_mode = getattr(sub_data, 'render_mode', 0x0004)
                        
                        if hasattr(sub_data, 'materials') and 'default_active' in sub_data.materials:
                            renderer.material.setup_from_dict(sub_data.materials['default_active'])
                                
                        obj_ent.add_child(child_ent, keep_world=False)
                        self.scene.add_entity(child_ent)
                    else:
                        renderer = obj_ent.add_component(MeshRenderer())
                        renderer.geometry = BufferObject(sub_data.vertices, sub_data.indices, vertex_size=sub_data.vertex_size)
                        renderer.geometry.pivot_offset = sub_data.pivot_offset
                        
                        renderer.geometry.filepath = path
                        renderer.geometry.name = getattr(sub_data, 'name', '')
                        renderer.geometry.render_mode = getattr(sub_data, 'render_mode', 0x0004)
                        
                        if hasattr(sub_data, 'materials') and 'default_active' in sub_data.materials:
                            renderer.material.setup_from_dict(sub_data.materials['default_active'])

            self.scene.selected_index = self.scene.entities.index(master_ent)
            
        except Exception as e:
            raise ResourceError(f"Failed to instantiate model hierarchy from '{path}'.\nReason: {e}")