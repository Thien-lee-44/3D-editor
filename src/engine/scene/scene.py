import re
from typing import List, Tuple

from src.engine.scene.entity import Entity
from src.engine.scene.components import TransformComponent, MeshRenderer, LightComponent, CameraComponent
from src.app.config import DEFAULT_MANIPULATION_MODE

class Scene:
    """
    Acts as the primary state container for the 3D world context, orchestrating all active entities.
    Employs Data-Oriented Design principles (Structural Caching) to optimize the render loop.
    Strictly decoupled from entity instantiation logic.
    """
    
    def __init__(self) -> None:
        self.entities: List[Entity] = []
        self.selected_index: int = -1 
        self.manipulation_mode: str = DEFAULT_MANIPULATION_MODE
        self.show_screen_axis: bool = True
        
        # Render caches for rapid access during the rendering loop
        self.cached_cameras: List[Tuple[TransformComponent, CameraComponent, Entity]] = []
        self.cached_lights: List[Tuple[TransformComponent, LightComponent, Entity]] = []
        self.cached_renderables: List[Tuple[TransformComponent, MeshRenderer, Entity]] = []

    def _rebuild_cache(self) -> None:
        """
        Categorizes and explicitly caches active components. 
        This is a deterministic function triggered exclusively whenever the scene topology mutates.
        """
        self.cached_cameras.clear()
        self.cached_lights.clear()
        self.cached_renderables.clear()
        
        for ent in self.entities:
            tf = ent.get_component(TransformComponent)
            if not tf: 
                continue
            
            cam = ent.get_component(CameraComponent)
            if cam: 
                self.cached_cameras.append((tf, cam, ent))
            
            light = ent.get_component(LightComponent)
            if light: 
                self.cached_lights.append((tf, light, ent))
            
            mesh = ent.get_component(MeshRenderer)
            if mesh: 
                self.cached_renderables.append((tf, mesh, ent))

    def _get_unique_name(self, desired_name: str) -> str:
        """Ensures entity names remain strictly unique within the hierarchy."""
        existing_names = {e.name for e in self.entities}
        if desired_name not in existing_names: 
            return desired_name
        
        base_name = desired_name
        counter = 1
        match = re.match(r"^(.*)\s\((\d+)\)$", desired_name)
        
        if match: 
            base_name = match.group(1)
            counter = int(match.group(2))
            
        while f"{base_name} ({counter})" in existing_names: 
            counter += 1
            
        return f"{base_name} ({counter})"

    def add_entity(self, entity: Entity) -> None:
        """Registers a newly instantiated entity into the global scene registry."""
        entity.name = self._get_unique_name(entity.name)
        self.entities.append(entity)
        self.selected_index = len(self.entities) - 1
        self._rebuild_cache()

    def remove_entity(self, index: int) -> None:
        """Safely executes the deletion lifecycle of an entity, recursively culling all attached children."""
        if 0 <= index < len(self.entities):
            ent = self.entities[index]
            
            if ent.parent: 
                ent.parent.remove_child(ent, keep_world=False)
                
            for child in list(ent.children): 
                if child in self.entities: 
                    self.remove_entity(self.entities.index(child))
                    
            if ent in self.entities: 
                self.entities.remove(ent)
            
            self.selected_index = -1
            self._rebuild_cache()

    def clear_entities(self) -> None:
        """Purges the entire scene state."""
        self.entities.clear()
        self.selected_index = -1
        self._rebuild_cache()