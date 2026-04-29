"""
Serialization Manager.
Handles scene state snapshots (Undo/Redo), project saving/loading (JSON), 
and routing to external geometry exporters.
"""

import json
from typing import Dict, Any, Optional

from src.engine.resources.resource_manager import ResourceManager
from src.engine.scene.entity import Entity
from src.engine.scene.components import TransformComponent, MeshRenderer, LightComponent, CameraComponent

class SerializationManager:
    """
    Manages the encoding and decoding of the Scene Graph and its entities 
    for disk persistence and application memory states.
    """

    def __init__(self, scene: Any, scene_mgr: Any) -> None:
        self.scene = scene
        self.scene_mgr = scene_mgr

    def get_scene_snapshot(self) -> str:
        """Generates a JSON string representing the entire current state of the scene."""
        entities_data = [self._serialize_entity(ent) for ent in self.scene.entities if ent.parent is None]
        return json.dumps(entities_data)

    def restore_snapshot(self, snapshot_str: str, current_aspect: float) -> None:
        """Reconstructs the scene graph from a previously serialized JSON string."""
        if not snapshot_str:
            return

        entities_data = json.loads(snapshot_str)
        self.scene.clear_entities()

        for ent_data in entities_data:
            ent = self._deserialize_entity(ent_data)
            self.scene_mgr._add_entity_recursive(ent)

    def save_project(self, file_path: str, metadata: Dict[str, Any]) -> None:
        """Compiles scene data, metadata, and asset manifests into a project file."""
        data = {
            "metadata": metadata,
            "assets": {
                "models": list(ResourceManager.project_models),
                "textures": list(ResourceManager.project_textures),
            },
            "entities": [self._serialize_entity(ent) for ent in self.scene.entities if ent.parent is None],
        }
        ResourceManager.save_project_file(file_path, data)

    def load_project(self, file_path: str, current_aspect: float) -> Dict[str, Any]:
        """
        Parses a project file, restores the asset manifest, and rebuilds the scene graph.
        Returns the parsed project metadata.
        """
        data = ResourceManager.load_project_file(file_path)
        self.scene.clear_entities()
        ResourceManager.clear_project_assets()

        for p in data.get("assets", {}).get("models", []):
            ResourceManager.add_project_model(p)

        for p in data.get("assets", {}).get("textures", []):
            ResourceManager.add_project_texture(p)

        for ent_data in data.get("entities", []):
            ent = self._deserialize_entity(ent_data)
            self.scene_mgr._add_entity_recursive(ent)

        return data.get("metadata", {})

    def export_scene_obj(self, export_dir: str) -> None:
        """Dispatches an export request to the OBJ exporter subsystem."""
        from src.engine.resources.exporter import OBJExporter

        top_level_entities = [ent for ent in self.scene.entities if ent.parent is None]
        OBJExporter.export(top_level_entities, export_dir)

    def _serialize_entity(self, ent: Entity) -> Dict[str, Any]:
        """Recursively encodes an entity and its attached components into a dictionary."""
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
        """Recursively decodes a dictionary payload and reconstructs an Entity and its components."""
        children_data = data.get("children", [])
        
        # Support legacy property keys
        is_group = data.get("is_group", data.get("isGroup", False))
        if not is_group and children_data:
            is_group = True

        ent = Entity(data.get("name", "Entity"), is_group=bool(is_group))
        
        raw_comps = data.get("components", {})
        comps = {str(k).lower(): v for k, v in raw_comps.items()} if isinstance(raw_comps, dict) else {}

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

        # Reconstruct child hierarchy
        for child_data in children_data:
            child_ent = self._deserialize_entity(child_data, ent)
            ent.add_child(child_ent, keep_world=False)

        return ent