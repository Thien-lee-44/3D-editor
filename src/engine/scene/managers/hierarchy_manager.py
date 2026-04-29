"""
Hierarchy Manager.

Manages entity relationships, parent-child structures, and grouping operations.
Ensures global transforms remain visually stable during reparenting.
"""

import glm
from typing import List, Dict, Optional, Any

from src.engine.scene.entity import Entity
from src.engine.scene.components import TransformComponent
from src.engine.scene.components.animation_cmp import AnimationComponent
from src.engine.scene.components.semantic_cmp import SemanticComponent


class HierarchyManager:
    """
    Coordinates modifications to the Scene Graph tree structure.
    Handles grouping, ungrouping, and synchronizing with the UI.
    """
    
    def __init__(self, scene: Any, scene_mgr: Any) -> None:
        self.scene = scene
        self.scene_mgr = scene_mgr

    def _invalidate_animation(self, entity: Entity) -> None:
        """Flushes animation keyframes when an entity undergoes structural hierarchy changes."""
        anim = entity.get_component(AnimationComponent)
        if anim:
            anim.keyframes.clear()
            if hasattr(anim, '_base_state_cache'):
                anim._base_state_cache.clear()

    def group_selected_entities(self, entity_ids: List[int]) -> None:
        """Creates a central group node and reparents selected entities under it."""
        if len(entity_ids) < 2:
            return
            
        valid_ents = [self.scene.entities[i] for i in entity_ids if 0 <= i < len(self.scene.entities)]
        top_level_ents = [e for e in valid_ents if e.parent not in valid_ents]
        
        if not top_level_ents: 
            return

        # Calculate geometric centroid for the group pivot
        centroid = glm.vec3(0.0)
        count = 0
        for ent in top_level_ents:
            tf = ent.get_component(TransformComponent)
            if tf:
                mat = tf.get_matrix()
                centroid += glm.vec3(mat[3][0], mat[3][1], mat[3][2])
                count += 1
                
        if count > 0:
            centroid /= count

        # Construct Group Entity
        group_ent = Entity("Group", is_group=True)
        group_tf = group_ent.add_component(TransformComponent())
        group_tf.position = centroid
        
        group_ent.add_component(AnimationComponent())
        # Default assignment to 'Misc' class
        group_ent.add_component(SemanticComponent(class_id=3))
        
        common_parent = top_level_ents[0].parent
        self.scene.add_entity(group_ent)
        
        if common_parent:
            common_parent.add_child(group_ent, keep_world=True)
            
        for ent in top_level_ents:
            if ent.parent:
                ent.parent.remove_child(ent, keep_world=True)
            group_ent.add_child(ent, keep_world=True)
            self._invalidate_animation(ent)
            
        self.scene.selected_index = self.scene.entities.index(group_ent)

    def ungroup_selected_entity(self) -> None:
        """Dissolves a group entity and elevates its children to its parent's level."""
        idx = self.scene.selected_index
        if idx < 0 or idx >= len(self.scene.entities): 
            return
            
        group_ent = self.scene.entities[idx]
        if not group_ent.is_group:
            return
            
        parent_ent = group_ent.parent
        children_snapshot = list(group_ent.children)
        
        for child in children_snapshot:
            group_ent.remove_child(child, keep_world=True)
            if parent_ent:
                parent_ent.add_child(child, keep_world=True)
            self._invalidate_animation(child)
                
        self.scene.remove_entity(idx)

    def sync_hierarchy_from_ui(self, hierarchy_mapping: Dict[int, Optional[int]]) -> None:
        """Synchronizes the backend entity tree with drag-and-drop actions from the UI."""
        for child_id, parent_id in hierarchy_mapping.items():
            if child_id >= len(self.scene.entities):
                continue
                
            child = self.scene.entities[child_id]
            current_parent = child.parent
            new_parent = self.scene.entities[parent_id] if parent_id is not None else None
            
            if current_parent != new_parent:
                if new_parent is not None and not new_parent.is_group:
                    continue

                if current_parent is not None:
                    current_parent.remove_child(child, keep_world=True)
                    
                if new_parent is not None:
                    new_parent.add_child(child, keep_world=True)
                    
                self._invalidate_animation(child)