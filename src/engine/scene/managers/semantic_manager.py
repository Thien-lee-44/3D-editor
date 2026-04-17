import random
from typing import Dict, Any, List

from src.engine.scene.entity import Entity
from src.engine.scene.components.semantic_cmp import SemanticComponent
from src.engine.synthetic.tracking_mgr import TrackingManager

class SemanticManager:
    """
    Handles Ground Truth semantic data generation, class ID management, 
    and recursive bounding-box inheritance for AI datasets.
    """
    def __init__(self, scene: Any) -> None:
        self.scene = scene
        self.semantic_classes: Dict[int, Dict[str, Any]] = {
            0: {"name": "Car", "color": [1.0, 0.0, 0.0]},           
            1: {"name": "Pedestrian", "color": [0.0, 1.0, 0.0]},    
            2: {"name": "Traffic Sign", "color": [0.0, 0.0, 1.0]},  
            3: {"name": "Misc", "color": [1.0, 1.0, 0.0]}           
        }

    def get_semantic_classes(self) -> Dict[int, Dict[str, Any]]:
        return self.semantic_classes

    def add_semantic_class(self, name: str) -> int:
        new_id = max(self.semantic_classes.keys()) + 1 if self.semantic_classes else 0
        r, g, b = random.random(), random.random(), random.random()
        self.semantic_classes[new_id] = {"name": name, "color": [r, g, b]}
        return new_id

    def update_semantic_class_color(self, class_id: int, color: List[float]) -> None:
        if class_id in self.semantic_classes:
            self.semantic_classes[class_id]["color"] = color

    def handle_semantic_property(self, ent: Entity, comp: SemanticComponent, prop: str, value: Any) -> None:
        if prop == "class_id":
            comp.class_id = int(value)
            if len(ent.children) > 0:
                self.propagate_semantics(ent, comp.class_id, comp.track_id)
            elif ent.parent is not None:
                comp.track_id = TrackingManager.get_next_id()

    def propagate_semantics(self, node: Entity, target_class: int, target_track: int) -> None:
        """Recursively forces all children to inherit the parent's semantic tags."""
        for child in node.children:
            c_sem = child.get_component(SemanticComponent)
            if not c_sem:
                c_sem = child.add_component(SemanticComponent())
                
            c_sem.class_id = target_class
            c_sem.track_id = target_track
            self.propagate_semantics(child, target_class, target_track)