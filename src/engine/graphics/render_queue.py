"""
Render Queue System.
Classifies and sorts renderable entities into distinct execution queues to optimize 
GPU state changes and resolve alpha-blending depth artifacts.
"""

import glm
from typing import List, Tuple, Any

class RenderQueue:
    """
    Manages the ordering of draw calls per frame.
    Separates opaque, transparent, and editor proxy objects to ensure correct 
    Painter's Algorithm execution for transparency and minimal context switching.
    """
    
    def __init__(self) -> None:
        # Queues store tuples of (TransformComponent, MeshRenderer, Entity)
        self.opaque: List[Tuple[Any, Any, Any]] = []
        self.transparent: List[Tuple[Any, Any, Any]] = []
        self.proxies: List[Tuple[Any, Any, Any]] = []

    def clear(self) -> None:
        """Flushes all queues from the previous frame."""
        self.opaque.clear()
        self.transparent.clear()
        self.proxies.clear()

    def build(self, scene: Any, camera_pos: glm.vec3) -> None:
        """
        Iterates over the Scene's cached renderables to populate and sort the draw queues.
        Requires the active camera position to calculate depth for transparent objects.
        """
        self.clear()
        
        if not scene:
            return

        transparent_with_dist: List[Tuple[float, Any, Any, Any]] = []

        for tf, mesh, ent in scene.cached_renderables:
            if not mesh.visible or not mesh.geometry:
                continue

            # Isolate Editor Proxies (UI icons for Cameras/Lights)
            if getattr(mesh, 'is_proxy', False):
                self.proxies.append((tf, mesh, ent))
                continue

            # Determine transparency status based on Opacity or forced Blend state
            mat = mesh.material
            is_transparent = False
            
            if mat:
                is_transparent = mat.opacity < 1.0 or getattr(mat.render_state, 'blend', False)

            if is_transparent:
                # Calculate distance to camera for depth sorting
                dist = glm.length(camera_pos - tf.global_position)
                transparent_with_dist.append((dist, tf, mesh, ent))
            else:
                self.opaque.append((tf, mesh, ent))

        # --- Sorting Strategy ---
        
        # 1. Opaque Queue: Sort by Material ID to minimize costly GPU context switches.
        self.opaque.sort(key=lambda item: id(item[1].material) if item[1].material else 0)

        # 2. Transparent Queue: Strict Back-to-Front sorting required for correct Alpha Blending.
        transparent_with_dist.sort(key=lambda item: item[0], reverse=True)
        
        # Strip the distance value, retaining only the functional tuple for rendering
        self.transparent = [(t[1], t[2], t[3]) for t in transparent_with_dist]