"""
Serialization Manager.

Handles JSON serialization, deserialization, project file I/O, 
state snapshots (Undo/Redo), and OBJ geometry exporting.
"""

import json
from typing import Dict, Any, Optional

from src.engine.resources.resource_manager import ResourceManager
from src.engine.scene.entity import Entity
from src.engine.scene.components import TransformComponent, MeshRenderer, LightComponent, CameraComponent
from src.engine.scene.components.animation_cmp import AnimationComponent
from src.engine.scene.components.semantic_cmp import SemanticComponent
from src.engine.scene.managers.semantic_manager import SemanticManager


class SerializationManager:
    """
    Encapsulates algorithms for converting the dynamic Scene Graph tree 
    into persistent storage formats and vice-versa.
    """
    
    def __init__(self, scene: Any, scene_mgr: Any) -> None:
        self.scene = scene
        self.scene_mgr = scene_mgr

    def get_scene_snapshot(self) -> str:
        """Generates a JSON string representing the current state for Undo/Redo tracking."""
        entities_data = [self._serialize_entity(ent) for ent in self.scene.entities if ent.parent is None]
        return json.dumps(entities_data)

    def restore_snapshot(self, snapshot_str: str, current_aspect: float) -> None:
        """Reconstructs the Scene Graph from a memory snapshot string."""
        if not snapshot_str: 
            return
            
        entities_data = json.loads(snapshot_str)
        self.scene.entities.clear()
        
        for ent_data in entities_data:
            ent = self._deserialize_entity(ent_data)
            self.scene_mgr._add_entity_recursive(ent)

        self._sync_cameras_aspect(current_aspect)

    def save_project(self, file_path: str, metadata: Dict[str, Any]) -> None:
        """Serializes the entire project including assets, semantics, and entities to disk."""
        data = {
            "metadata": metadata,
            "semantic_classes": self.scene_mgr.semantic.semantic_classes,
            "assets": {
                "models": list(ResourceManager.project_models),
                "textures": list(ResourceManager.project_textures)
            },
            "entities": [self._serialize_entity(ent) for ent in self.scene.entities if ent.parent is None]
        }
        ResourceManager.save_project_file(file_path, data)

    def load_project(self, file_path: str, current_aspect: float) -> Dict[str, Any]:
        """Loads and reconstructs a complete project environment from a JSON file."""
        data = ResourceManager.load_project_file(file_path)

        self.scene.entities.clear()
        ResourceManager.clear_project_assets()
        
        loaded_classes = data.get("semantic_classes", SemanticManager.DEFAULT_SEMANTIC_CLASSES)
        
        self.scene_mgr.semantic.semantic_classes = {}
        for k, v in loaded_classes.items():
            if isinstance(v, str): 
                self.scene_mgr.semantic.semantic_classes[int(k)] = {"name": v, "color": [0.8, 0.8, 0.8]}
            else:
                self.scene_mgr.semantic.semantic_classes[int(k)] = v
        
        for p in data.get("assets", {}).get("models", []): 
            ResourceManager.add_project_model(p)
            
        for p in data.get("assets", {}).get("textures", []): 
            ResourceManager.add_project_texture(p)
        
        for ent_data in data.get("entities", []):
            ent = self._deserialize_entity(ent_data)
            self.scene_mgr._add_entity_recursive(ent)
    
        self._sync_cameras_aspect(current_aspect)
            
        return data.get("metadata", {})

    def _sync_cameras_aspect(self, current_aspect: float) -> None:
        """
        Recursively forces all Camera components to adopt the current UI viewport aspect ratio.
        Prevents visual distortion (stretching/squashing) immediately after loading a project.
        """
        def update_recursive(entity: Entity) -> None:
            cam = entity.get_component(CameraComponent)
            if cam:
                if hasattr(cam, 'aspect_ratio'):
                    cam.aspect_ratio = current_aspect
                else:
                    cam.aspect = current_aspect
                if hasattr(cam, 'update_projection_matrix'):
                    cam.update_projection_matrix()
                    
            for child in entity.children:
                update_recursive(child)
                
        for ent in self.scene.entities:
            if ent.parent is None:
                update_recursive(ent)

    def export_scene_obj(self, export_dir: str) -> None:
        """Delegates geometric exporting to the OBJ format via the ResourceManager."""
        from src.engine.resources.exporter import OBJExporter
        top_level_entities = [ent for ent in self.scene.entities if ent.parent is None]
        OBJExporter.export(top_level_entities, export_dir)

    def _serialize_entity(self, ent: Entity) -> Dict[str, Any]:
        """Serializes a single entity and its components recursively."""
        data = {"name": ent.name, "is_group": ent.is_group, "components": {}, "children": []}
        
        tf = ent.get_component(TransformComponent)
        if tf: data["components"]["transform"] = tf.to_dict()
            
        mesh = ent.get_component(MeshRenderer)
        if mesh: data["components"]["mesh"] = mesh.to_dict()
            
        light = ent.get_component(LightComponent)
        if light: data["components"]["light"] = light.to_dict()
            
        cam = ent.get_component(CameraComponent)
        if cam: data["components"]["camera"] = cam.to_dict()
            
        anim = ent.get_component(AnimationComponent)
        if anim and hasattr(anim, 'serialize'): 
            data["components"]["animation"] = anim.serialize()
            
        semantic = ent.get_component(SemanticComponent)
        if semantic and hasattr(semantic, 'serialize'): 
            data["components"]["semantic"] = semantic.serialize()
            
        for child in ent.children: 
            data["children"].append(self._serialize_entity(child))
            
        return data

    def _deserialize_entity(self, data: Dict[str, Any], parent: Optional[Entity] = None) -> Entity:
        """Reconstructs an entity and instantiates its components from dictionary data."""
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
            
        if "animation" in comps:
            anim = ent.add_component(AnimationComponent())
            if hasattr(anim, 'deserialize'): 
                anim.deserialize(comps["animation"])
                
        if "semantic" in comps:
            semantic = ent.add_component(SemanticComponent())
            if hasattr(semantic, 'deserialize'): 
                semantic.deserialize(comps["semantic"])
                
        for child_data in data.get("children", []):
            child_ent = self._deserialize_entity(child_data, ent) 
            ent.add_child(child_ent, keep_world=False)
            
        return ent