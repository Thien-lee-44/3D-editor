"""
Render Queue Management.

Sorts renderable entities into distinct execution queues (Opaque, Transparent, Proxies)
to optimize GPU state changes and resolve Alpha Blending artifacts.
"""

import glm
from typing import List, Tuple, Any


class RenderQueue:
    """
    Classifies and sorts renderables to ensure optimal execution order during the draw loop.
    """
    
    def __init__(self) -> None:
        self.opaque: List[Tuple[Any, Any, Any]] = []
        self.transparent: List[Tuple[Any, Any, Any]] = []
        self.proxies: List[Tuple[Any, Any, Any]] = []

    def clear(self) -> None:
        """Flushes the queues from the previous frame."""
        self.opaque.clear()
        self.transparent.clear()
        self.proxies.clear()

    def build(self, scene: Any, camera_pos: glm.vec3) -> None:
        """
        Populates draw queues from the Scene cache. Calculates depths for Painter's Algorithm.
        """
        self.clear()
        if not scene:
            return

        transparent_with_dist = []

        for tf, mesh, ent in scene.cached_renderables:
            if not mesh.visible or not mesh.geometry:
                continue

            if getattr(mesh, 'is_proxy', False):
                self.proxies.append((tf, mesh, ent))
                continue

            mat = mesh.material
            is_transparent = False
            if mat:
                is_transparent = mat.opacity < 1.0 or getattr(mat.render_state, 'blend', False)

            if is_transparent:
                dist = glm.length(camera_pos - tf.global_position)
                transparent_with_dist.append((dist, tf, mesh, ent))
            else:
                self.opaque.append((tf, mesh, ent))

        # 1. Opaque Queue: Sort by Material ID to minimize costly GPU context switches
        self.opaque.sort(key=lambda item: id(item[1].material) if item[1].material else 0)

        # 2. Transparent Queue: Strict Back-to-Front sorting (Painter's Algorithm)
        transparent_with_dist.sort(key=lambda item: item[0], reverse=True)
        self.transparent = [(t[1], t[2], t[3]) for t in transparent_with_dist]