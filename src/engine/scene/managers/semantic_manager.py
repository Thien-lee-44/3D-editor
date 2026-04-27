import random
from typing import Dict, Any, List, Optional
from src.engine.scene.entity import Entity
from src.engine.scene.components.semantic_cmp import SemanticComponent

class SemanticManager:
    """
    Sub-manager handling AI labeling logic.
    Forces strict recursive propagation for unified objects and respects 
    user-defined bulk-assignment rules for organizational folders.
    """
    def __init__(self, scene: Any) -> None:
        self.scene = scene
        self.semantic_classes: Dict[int, Dict[str, Any]] = {
            0: {"name": "Car", "color": [1.0, 0.0, 0.0]},
            1: {"name": "Pedestrian", "color": [0.0, 1.0, 0.0]},
            2: {"name": "Traffic Sign", "color": [0.0, 0.0, 1.0]},
            3: {"name": "Misc", "color": [1.0, 1.0, 0.0]}
        }
        self._next_class_id: int = len(self.semantic_classes)

    def handle_semantic_property(self, ent: Entity, comp: SemanticComponent, prop: str, value: Any) -> None:
        """Processes mutations and enforces hierarchical rules."""
        if prop == "class_id":
            comp.class_id = int(value)
            
            force_prop = getattr(comp, "is_merged_instance", True) or getattr(comp, "propagate_to_children", True)
            if force_prop:
                self._propagate_class_id(ent, comp.class_id)
            
        elif prop == "is_merged_instance":
            comp.is_merged_instance = bool(value)
            self._invalidate_hierarchy_tracking(ent)
            
            if comp.is_merged_instance:
                self._propagate_class_id(ent, comp.class_id)

        elif prop == "propagate_to_children":
            comp.propagate_to_children = bool(value)
            
            if comp.propagate_to_children and not getattr(comp, "is_merged_instance", True):
                self._propagate_class_id(ent, comp.class_id)

    def _propagate_class_id(self, node: Entity, class_id: int) -> None:
        """Recursively forces downstream children to match the parent's semantic class."""
        for child in node.children:
            c_sem = child.get_component(SemanticComponent)
            if not c_sem:
                c_sem = child.add_component(SemanticComponent())
                
            c_sem.class_id = class_id
            self._propagate_class_id(child, class_id)

    def _invalidate_hierarchy_tracking(self, node: Entity) -> None:
        """Resets tracking states to force dynamic re-evaluation during export."""
        c_sem = node.get_component(SemanticComponent)
        if c_sem:
            c_sem.track_id = -1
            
        for child in node.children:
            self._invalidate_hierarchy_tracking(child)

    def get_semantic_classes(self) -> Dict[int, Dict[str, Any]]:
        return self.semantic_classes

    def add_semantic_class(self, name: str) -> int:
        new_id = self._next_class_id
        self._next_class_id += 1
        
        self.semantic_classes[new_id] = {
            "name": name, 
            "color": [random.uniform(0.1, 1.0) for _ in range(3)]
        }
        return new_id

    def update_semantic_class_color(self, class_id: int, color: List[float]) -> None:
        if class_id in self.semantic_classes:
            self.semantic_classes[class_id]["color"] = color
            
    def remove_semantic_class(self, class_id: int) -> None:
        if class_id == 0: return 
        if class_id in self.semantic_classes:
            del self.semantic_classes[class_id]
            for ent in self.scene.entities:
                comp = ent.get_component(SemanticComponent)
                if comp and comp.class_id == class_id:
                    comp.class_id = 0