"""
Clipboard Manager.
Handles copy, cut, paste, and deletion operations for scene entities.
"""

import copy
import glm
from typing import Any, Optional

from src.engine.scene.entity import Entity
from src.engine.scene.components import TransformComponent, LightComponent
from src.app.exceptions import SimulationError
from src.app.config import MAX_LIGHTS, PASTE_OFFSET

class ClipboardManager:
    """
    Manages the entity clipboard and ensures logical constraints 
    (such as maximum light counts) are respected during duplication.
    """

    def __init__(self, scene: Any, scene_mgr: Any) -> None:
        self.scene = scene
        self.scene_mgr = scene_mgr
        self.clipboard: Optional[Entity] = None

    def has_clipboard(self) -> bool:
        """Checks if there is a valid entity currently stored in the clipboard."""
        return self.clipboard is not None

    def copy_selected(self) -> None:
        """Deep clones the currently selected entity into the clipboard."""
        if self.scene.selected_index >= 0:
            self.clipboard = copy.deepcopy(self.scene.entities[self.scene.selected_index])

    def cut_selected(self) -> None:
        """Copies the selected entity and immediately removes it from the scene."""
        self.copy_selected()
        self.delete_selected()

    def paste_copied(self) -> None:
        """
        Instantiates a duplicate of the clipboard entity into the scene.
        Enforces domain-specific constraints (e.g., hardware lighting limits) 
        and applies a spatial offset to prevent visual overlap.
        """
        if not self.clipboard:
            raise SimulationError("Cannot paste: The clipboard is currently empty.")

        # Enforce maximum active light limitations
        light = self.clipboard.get_component(LightComponent)
        if light:
            ltype = light.type
            count = sum(
                1 for e in self.scene.entities
                if e.get_component(LightComponent) and e.get_component(LightComponent).type == ltype
            )
            limit = MAX_LIGHTS.get(ltype, 0)
            if count >= limit:
                raise SimulationError(f"Cannot paste {ltype} Light. Limit of {limit} reached.")

        new_ent = copy.deepcopy(self.clipboard)
        new_ent.name += " (Copy)"

        # Apply spatial offset for visibility
        tf = new_ent.get_component(TransformComponent)
        if tf:
            tf.position += glm.vec3(*PASTE_OFFSET)

        self.scene_mgr._add_entity_recursive(new_ent)

    def delete_selected(self) -> None:
        """Removes the currently selected entity from the scene graph."""
        if self.scene.selected_index >= 0:
            self.scene.remove_entity(self.scene.selected_index)