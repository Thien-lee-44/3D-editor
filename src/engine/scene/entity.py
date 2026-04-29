"""
Entity Component System (ECS) Core.

Defines the fundamental building blocks of the 3D scene: 
Entities (Node containers) and Components (Data/Logic payloads).
"""

import glm
import copy
from typing import List, Optional, Type, TypeVar

from src.app.config import DEFAULT_ENTITY_NAME

T = TypeVar('T', bound='Component')


class Component:
    """
    Abstract base class for all functional modules in the Entity-Component System (ECS).
    Components strictly hold data and specific logic, while Entities act as generic containers.
    """
    def __init__(self) -> None:
        self.entity: Optional['Entity'] = None 


class Entity:
    """
    Represents a discrete hierarchical node within the 3D Scene Graph.
    Acts as a container for Components and manages parent-child relationships 
    to propagate spatial transformations downstream.
    """
    def __init__(self, name: str = DEFAULT_ENTITY_NAME, is_group: bool = False) -> None:
        self.name: str = name
        self.is_group: bool = is_group
        self.components: List[Component] = []
        self.parent: Optional['Entity'] = None     
        self.children: List['Entity'] = []     
        
    def add_component(self, component: Component) -> Component:
        """Binds a component instance to this entity and establishes the reverse structural linkage."""
        component.entity = self
        self.components.append(component)
        return component
        
    def get_component(self, comp_type: Type[T]) -> Optional[T]:
        """
        Scans attached components and returns the first match of the requested type.
        Utilizes type generics for seamless IDE autocomplete support.
        """
        for c in self.components:
            if isinstance(c, comp_type): 
                return c
        return None

    def add_child(self, child: 'Entity', keep_world: bool = False) -> None:
        """
        Establishes a parent-child relationship between this entity and the specified child.
        If keep_world is True, the child's local transform is mathematically recalculated 
        to ensure its absolute visual position in the 3D world remains identical.
        """
        if not self.is_group:
            return 
        
        # Local import to prevent circular dependencies at boot time
        from src.engine.scene.components import TransformComponent
        
        tf: Optional[TransformComponent] = None
        world_mat = glm.mat4(1.0)
        
        if keep_world:
            tf = child.get_component(TransformComponent)
            if tf:
                world_mat = tf.get_matrix()
        
        if child.parent: 
            child.parent.remove_child(child, keep_world=False)
            
        child.parent = self
        if child not in self.children: 
            self.children.append(child)
            
        if keep_world and tf:
            parent_tf = self.get_component(TransformComponent)
            if parent_tf:
                parent_world = parent_tf.get_matrix()
                tf.set_from_matrix(glm.inverse(parent_world) * world_mat)
            else:
                tf.set_from_matrix(world_mat)

    def remove_child(self, child: 'Entity', keep_world: bool = False) -> None:
        """
        Severs the hierarchical link between this entity and the specified child.
        Can optionally preserve the child's absolute world-space transform.
        """
        if child in self.children:
            from src.engine.scene.components import TransformComponent
            
            tf: Optional[TransformComponent] = None
            world_mat = glm.mat4(1.0)
            
            if keep_world:
                tf = child.get_component(TransformComponent)
                if tf:
                    world_mat = tf.get_matrix()
            
            self.children.remove(child)
            child.parent = None
            
            if keep_world and tf:
                tf.set_from_matrix(world_mat)

    def __deepcopy__(self, memo: dict) -> 'Entity':
        """
        Custom recursive deep cloning algorithm. 
        Detects the copy operation root dynamically via the memoization dictionary to extract
        pure global transformations by traversing the parent hierarchy, maintaining exact world bounds.
        """
        new_ent = type(self)(self.name, self.is_group) 
        memo[id(self)] = new_ent
        
        for comp in self.components:
            new_ent.add_component(copy.deepcopy(comp, memo))
            
        new_ent.parent = None
        
        is_copy_root = self.parent is None or id(self.parent) not in memo
        
        if is_copy_root and self.parent is not None:
            from src.engine.scene.components import TransformComponent
            orig_tf = self.get_component(TransformComponent)
            new_tf = new_ent.get_component(TransformComponent)
            
            if orig_tf and new_tf:
                g_scl = glm.vec3(1.0)
                curr: Optional['Entity'] = self
                while curr:
                    curr_tf = curr.get_component(TransformComponent)
                    if curr_tf:
                        g_scl *= curr_tf.scale
                    curr = curr.parent
                    
                if not new_tf.locked_axes.get("pos", False):
                    new_tf.position = glm.vec3(getattr(orig_tf, 'global_position', orig_tf.position))
                    
                if not new_tf.locked_axes.get("rot", False):
                    new_tf.quat_rot = glm.quat(getattr(orig_tf, 'global_quat_rot', orig_tf.quat_rot))
                    new_tf.rotation = glm.degrees(glm.eulerAngles(new_tf.quat_rot))
                    
                if not new_tf.locked_axes.get("scl", False):
                    new_tf.scale = g_scl
                
                new_tf.is_dirty = True
                if hasattr(new_tf, 'sync_from_gui'):
                    new_tf.sync_from_gui()
        
        for child in self.children:
            new_child = copy.deepcopy(child, memo)
            new_ent.add_child(new_child, keep_world=False) 
            
        return new_ent