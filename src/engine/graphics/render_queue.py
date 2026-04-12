import glm
from typing import List, Tuple, Any

class RenderQueue:
    """
    Iterates through the Scene, classifying and sorting renderable entities into distinct execution queues.
    This architecture optimizes GPU state changes (minimizing context switching) 
    and resolves Alpha Blending depth-sorting artifacts.
    """
    
    def __init__(self) -> None:
        # Stores tuples of (TransformComponent, MeshRenderer, Entity)
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
        Iterates over the Scene's cached renderables to populate and sort the draw queues.
        Requires the active camera position to calculate depth for transparent objects.
        """
        self.clear()
        
        if not scene:
            return

        transparent_with_dist = []

        for tf, mesh, ent in scene.cached_renderables:
            # Skip invisible meshes or those lacking geometric data
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

        # ==========================================
        # SORTING STRATEGY
        # ==========================================
        
        # 1. Opaque Queue: Sort by Material ID to minimize costly GPU context switches (Shader/Texture re-binding).
        self.opaque.sort(key=lambda item: id(item[1].material) if item[1].material else 0)

        # 2. Transparent Queue: Strict Back-to-Front sorting (Painter's Algorithm) required for correct Alpha Blending.
        transparent_with_dist.sort(key=lambda item: item[0], reverse=True)
        
        # Strip the distance value, retaining only the functional tuple
        self.transparent = [(t[1], t[2], t[3]) for t in transparent_with_dist]