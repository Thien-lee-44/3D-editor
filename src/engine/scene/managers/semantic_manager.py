import random
from typing import Dict, Any, List, Optional

from src.engine.scene.entity import Entity
from src.engine.scene.components.semantic_cmp import SemanticComponent
from src.engine.synthetic.tracking_mgr import TrackingManager

class SemanticManager:
    """
    Sub-manager responsible for handling AI/CV Ground Truth labels.
    
    Manages the global dictionary of semantic classes (e.g., Car, Pedestrian), 
    handles automatic color generation, and enforces recursive semantic 
    inheritance (Class ID and Track ID) across hierarchical 3D entity structures.
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

    def get_semantic_classes(self) -> Dict[int, Dict[str, Any]]:
        return self.semantic_classes

    def add_semantic_class(self, name: str) -> int:
        new_id = self._next_class_id
        self._next_class_id += 1
        
        r = random.uniform(0.1, 1.0)
        g = random.uniform(0.1, 1.0)
        b = random.uniform(0.1, 1.0)
        
        self.semantic_classes[new_id] = {"name": name, "color": [r, g, b]}
        return new_id

    def update_semantic_class_color(self, class_id: int, color: List[float]) -> None:
        if class_id in self.semantic_classes:
            self.semantic_classes[class_id]["color"] = color

    def handle_semantic_property(self, ent: Entity, comp: SemanticComponent, prop: str, value: Any) -> None:
        """
        Processes semantic mutations initiated by the Editor UI.
        Includes a special 'INHERIT' protocol to dynamically merge orphaned children back into their parent group.
        """
        if prop == "class_id":
            comp.class_id = int(value)
            
            if len(ent.children) > 0:
                self._propagate_semantics(ent, class_id=comp.class_id, track_id=comp.track_id)
            elif ent.parent is not None:
                comp.track_id = TrackingManager.get_next_id()
                
        elif prop == "track_id":
            # [NEW FEATURE]: Automatic Parent Re-merging
            if str(value) == "INHERIT":
                parent = ent.parent
                while parent:
                    p_sem = parent.get_component(SemanticComponent)
                    if p_sem:
                        # Adopt Parent's identity
                        comp.track_id = p_sem.track_id
                        comp.class_id = p_sem.class_id
                        # Cascade adoption down to this entity's children if any
                        if len(ent.children) > 0:
                            self._propagate_semantics(ent, class_id=comp.class_id, track_id=comp.track_id)
                        break
                    parent = parent.parent
            else:
                comp.track_id = int(value)
                if len(ent.children) > 0:
                    self._propagate_semantics(ent, track_id=comp.track_id)

    def _propagate_semantics(self, node: Entity, class_id: Optional[int] = None, track_id: Optional[int] = None) -> None:
        """Recursively forces all downstream children to inherit the specified semantic tags."""
        for child in node.children:
            c_sem = child.get_component(SemanticComponent)
            if not c_sem:
                c_sem = child.add_component(SemanticComponent())
                
            if class_id is not None:
                c_sem.class_id = class_id
            if track_id is not None:
                c_sem.track_id = track_id
                
            self._propagate_semantics(child, class_id, track_id)