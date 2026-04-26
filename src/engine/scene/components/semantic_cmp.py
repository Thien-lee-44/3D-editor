from typing import Dict, Any

class SemanticComponent:
    """
    Stores Ground Truth semantic data for AI dataset generation.
    Includes boundary flags and propagation rules for hierarchical evaluation.
    """
    def __init__(self, track_id: int = -1, class_id: int = 0) -> None:
        self.track_id: int = track_id
        self.class_id: int = class_id  
        
        self.is_merged_instance: bool = True
        self.propagate_to_children: bool = True
        
        self.entity = None

    def serialize(self) -> Dict[str, Any]:
        """Packages semantic data into a UI-agnostic payload."""
        is_group = getattr(self.entity, "is_group", False) if self.entity else False
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "is_merged_instance": getattr(self, "is_merged_instance", True),
            "propagate_to_children": getattr(self, "propagate_to_children", True),
            "is_group": is_group
        }

    def deserialize(self, data: Dict[str, Any]) -> None:
        """Reconstructs component state from JSON."""
        self.track_id = data.get("track_id", -1)
        self.class_id = data.get("class_id", 0)
        self.is_merged_instance = data.get("is_merged_instance", True)
        self.propagate_to_children = data.get("propagate_to_children", True)