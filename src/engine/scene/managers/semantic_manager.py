"""
Semantic Manager.

Sub-manager handling AI labeling logic.
Forces strict recursive propagation for unified objects and respects 
user-defined bulk-assignment rules for organizational folders.
"""

import random
from typing import Dict, Any, List

from src.engine.scene.entity import Entity
from src.engine.scene.components.semantic_cmp import SemanticComponent


class SemanticManager:
    """
    Manages global semantic classes and enforces inheritance rules 
    for ground truth annotation across the Scene Graph.
    """
    
    DEFAULT_SEMANTIC_CLASSES = {
        0: {"name": "Car", "color": [1.0, 0.0, 0.0]},
        1: {"name": "Pedestrian", "color": [0.0, 1.0, 0.0]},
        2: {"name": "Traffic Sign", "color": [0.0, 0.0, 1.0]},
        3: {"name": "Misc", "color": [1.0, 1.0, 0.0]}
    }

    def __init__(self, scene: Any) -> None:
        self.scene = scene
        self.semantic_classes: Dict[int, Dict[str, Any]] = dict(self.DEFAULT_SEMANTIC_CLASSES)
        self._next_class_id: int = len(self.semantic_classes)

    def handle_semantic_property(self, ent: Entity, comp: SemanticComponent, prop: str, value: Any) -> None:
        """Processes property mutations and enforces hierarchical semantic rules."""
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
        """Resets tracking states to force dynamic re-evaluation during data export."""
        c_sem = node.get_component(SemanticComponent)
        if c_sem:
            c_sem.track_id = -1
            
        for child in node.children:
            self._invalidate_hierarchy_tracking(child)

    def get_semantic_classes(self) -> Dict[int, Dict[str, Any]]:
        """Retrieves the active dictionary of semantic classes."""
        return self.semantic_classes

    def add_semantic_class(self, name: str) -> int:
        """Registers a new semantic class. Ensures contiguous ID assignment."""
        new_id = len(self.semantic_classes)
        
        self.semantic_classes[new_id] = {
            "name": name, 
            "color": [random.uniform(0.1, 1.0) for _ in range(3)]
        }
        
        self._next_class_id = len(self.semantic_classes)
        return new_id

    def update_semantic_class_color(self, class_id: int, color: List[float]) -> None:
        """Updates the RGB representation color for a specific class ID."""
        if class_id in self.semantic_classes:
            self.semantic_classes[class_id]["color"] = color
            
    def remove_semantic_class(self, class_id: int) -> None:
        """Deletes a custom class, reverts associated entities, and re-indexes to maintain contiguous IDs."""
        if class_id == 0: 
            return 
            
        if class_id in self.semantic_classes:
            del self.semantic_classes[class_id]
            
            for ent in self.scene.entities:
                comp = ent.get_component(SemanticComponent)
                if comp and comp.class_id == class_id:
                    comp.class_id = 0
            
            self._reindex_classes()

    def _reindex_classes(self) -> None:
        """Internal helper to ensure class IDs remain strictly sequential after deletion."""
        old_classes = self.semantic_classes.copy()
        self.semantic_classes.clear()
        
        id_mapping = {} 
        new_id = 0
        
        for old_id in sorted(old_classes.keys()):
            self.semantic_classes[new_id] = old_classes[old_id]
            id_mapping[old_id] = new_id
            new_id += 1
            
        self._next_class_id = len(self.semantic_classes)
        
        for ent in self.scene.entities:
            comp = ent.get_component(SemanticComponent)
            if comp:
                if comp.class_id in id_mapping:
                    comp.class_id = id_mapping[comp.class_id]
                else:
                    comp.class_id = 0