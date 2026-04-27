import glm
import os
from OpenGL.GL import *
from typing import Any, Dict, Optional, List, Tuple, Union

from src.engine.scene.scene import Scene
from src.engine.graphics.editor_renderer import Renderer, GizmoRenderer, HUDRenderer
from src.engine.graphics.synthetic_renderer import SyntheticRenderer
from src.engine.resources.resource_manager import ResourceManager
from src.engine.geometry.primitives import PrimitivesManager
from src.engine.scene.entity_factory import EntityFactory
from src.engine.scene.scene_manager import SceneManager
from src.engine.core.interaction_manager import InteractionManager
from src.engine.scene.animator import AnimatorSystem

class Engine:
    def __init__(self) -> None:
        self.scene = None
        self.renderer = None
        self.synthetic_renderer = None
        self.gizmo_renderer = None
        self.hud_renderer = None
        self.entity_fac = None
        self.scene_mgr = None
        self.interaction_mgr = None
        self.animator = None

    def init_viewport_gl(self) -> None:
        self.scene = Scene()
        
        self.renderer = Renderer()
        self.synthetic_renderer = SyntheticRenderer()
        self.gizmo_renderer = GizmoRenderer()
        
        self.entity_fac = EntityFactory(self.scene)
        self.scene_mgr = SceneManager(self.scene)
        self.interaction_mgr = InteractionManager(self.scene)
        self.animator = AnimatorSystem(self.scene)
        
        self.entity_fac.setup_default_scene()
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.0, 0.0, 0.0, 1.0)

    def init_hud_gl(self) -> None:
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.15, 0.15, 0.15, 1.0)
        self.hud_renderer = HUDRenderer()

    def _sync_active_camera_aspect(self, w: int, h: int) -> None:
        if self.interaction_mgr:
            _, cam = self.interaction_mgr._get_active_camera()
            if cam:
                cam.aspect = w / max(h, 1)

    def resize_gl(self, w: int, h: int) -> None:
        if not self.scene: 
            return
            
        glViewport(0, 0, w, h)
        self._sync_active_camera_aspect(w, h)

    def render_viewport(self, w: int, h: int, bg_color: tuple, active_axis: str, hovered_axis: str, hovered_screen_axis: str) -> None:
        if not self.scene or not self.renderer or not self.interaction_mgr: 
            return
            
        self._sync_active_camera_aspect(w, h)
            
        glClearColor(*bg_color, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        self.renderer.render_scene(self.scene, w, h)
        
        cam_tf, cam = self.interaction_mgr._get_active_camera()
        
        if cam_tf and self.gizmo_renderer: 
            self.gizmo_renderer.render(
                self.scene, 
                cam, 
                cam_tf, 
                w, 
                h, 
                active_axis or hovered_axis
            )

    def capture_fbo_frame(self, width: int, height: int, mode: str = "RGB", return_texture_id: bool = False) -> Union[bytes, int]:
        if not self.synthetic_renderer or not self.scene:
            return 0 if return_texture_id else b""
        self._sync_active_camera_aspect(width, height)
        return self.synthetic_renderer.capture_fbo_frame(self.scene, width, height, mode, return_texture_id)

    def raycast_select(self, mx: float, my: float, width: int, height: int) -> int:
        if not self.renderer or not self.scene:
            return -1
        self._sync_active_camera_aspect(width, height)
        return self.renderer.raycast_select(self.scene, mx, my, width, height)

    def render_sun_hud(self, w: int, h: int, active_axis: str, is_hover: bool) -> None:
        if not self.scene or self.scene.selected_index < 0 or not self.hud_renderer or not self.interaction_mgr: 
            return
            
        self._sync_active_camera_aspect(w, h)
            
        from src.engine.scene.components import TransformComponent
        target_tf = self.scene.entities[self.scene.selected_index].get_component(TransformComponent)
        cam_tf, cam = self.interaction_mgr._get_active_camera()
        
        view = glm.mat4(glm.mat3(cam.get_view_matrix())) if cam and cam_tf else glm.mat4(1.0)
        
        if target_tf: 
            self.hud_renderer.render(w, h, active_axis, is_hover, target_tf, view)

    def preload_model_to_cache(self, path: str) -> None:
        ResourceManager.get_model(path)
    
    def get_project_models(self) -> list: 
        return list(ResourceManager.project_models)
        
    def get_project_textures(self) -> list: 
        return list(ResourceManager.project_textures)
        
    def import_project_model(self, path: str) -> None: 
        ResourceManager.add_project_model(path)
        
    def import_project_texture(self, path: str) -> None: 
        ResourceManager.add_project_texture(path)
        
    def get_3d_primitive_names(self) -> list: 
        return list(PrimitivesManager.get_3d_paths().keys())
        
    def get_2d_primitive_names(self) -> list: 
        return list(PrimitivesManager.get_2d_paths().keys())

    def auto_load_default_assets(self, tex_dir: str) -> None:
        os.makedirs(tex_dir, exist_ok=True)
        for f in os.listdir(tex_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')): 
                ResourceManager.add_project_texture(os.path.join(tex_dir, f).replace('\\', '/'))

    def delete_project_asset(self, path: str, asset_type: str) -> None:
        if asset_type == 'TEXTURE' and path in ResourceManager.project_textures: 
            ResourceManager.project_textures.remove(path)
        elif asset_type == 'MODEL' and path in ResourceManager.project_models: 
            ResourceManager.project_models.remove(path)

    def has_clipboard(self) -> bool:
        return self.scene_mgr.has_clipboard() if self.scene_mgr else False

    def get_selected_entity_id(self) -> int:
        return self.scene_mgr.get_selected_entity_id() if self.scene_mgr else -1

    def select_entity(self, idx: int) -> None:
        if self.scene_mgr: self.scene_mgr.select_entity(idx)

    def set_manipulation_mode(self, mode: str) -> None:
        if self.scene_mgr: self.scene_mgr.set_manipulation_mode(mode)

    def get_scene_entities_list(self) -> List[Dict[str, Any]]:
        return self.scene_mgr.get_scene_entities_list() if self.scene_mgr else []

    def reset_entity_transform(self, index: int) -> None:
        if self.scene_mgr: self.scene_mgr.reset_entity_transform(index)

    def update_light_direction(self, yaw: float, pitch: float) -> None:
        if self.scene_mgr: self.scene_mgr.update_light_direction(yaw, pitch)

    def set_active_camera_selected(self) -> None:
        if self.scene_mgr: self.scene_mgr.set_active_camera_selected()

    def get_selected_transform_state(self) -> Optional[Tuple[str, Tuple[float, float, float]]]:
        return self.scene_mgr.get_selected_transform_state() if self.scene_mgr else None

    def clear_scene(self) -> None:
        if self.scene_mgr: self.scene_mgr.clear_scene()
        if self.entity_fac: self.entity_fac.setup_default_scene()

    def get_selected_entity_data(self) -> Optional[Dict[str, Any]]:
        return self.scene_mgr.get_selected_entity_data() if self.scene_mgr else None

    def set_component_property(self, comp_name: str, prop: str, value: Any) -> None:
        if self.scene_mgr: self.scene_mgr.set_component_property(comp_name, prop, value)
        
    def set_component_properties(self, comp_name: str, payload: Dict[str, Any]) -> None:
        if self.scene_mgr: self.scene_mgr.set_component_properties(comp_name, payload)
        
    def group_selected_entities(self, entity_ids: List[int]) -> None:
        if self.scene_mgr: self.scene_mgr.group_selected_entities(entity_ids)

    def ungroup_selected_entity(self) -> None:
        if self.scene_mgr: self.scene_mgr.ungroup_selected_entity()

    def copy_selected(self) -> None:
        if self.scene_mgr: self.scene_mgr.copy_selected()

    def cut_selected(self) -> None:
        if self.scene_mgr: self.scene_mgr.cut_selected()

    def paste_copied(self) -> None:
        if self.scene_mgr: self.scene_mgr.paste_copied()

    def delete_selected(self) -> None:
        if self.scene_mgr: self.scene_mgr.delete_selected()

    def toggle_visibility_selected(self) -> None:
        if self.scene_mgr: self.scene_mgr.toggle_visibility_selected()

    def toggle_all_lights(self, is_on: bool) -> None:
        if self.scene_mgr: self.scene_mgr.toggle_all_lights(is_on)

    def toggle_all_proxies(self, is_visible: bool) -> None:
        if self.scene_mgr: self.scene_mgr.toggle_all_proxies(is_visible)

    def sync_hierarchy_from_ui(self, hierarchy_mapping: Dict[int, Optional[int]]) -> None:
        if self.scene_mgr: self.scene_mgr.sync_hierarchy_from_ui(hierarchy_mapping)

    def load_texture_to_selected(self, map_attr: str, filepath: str) -> None:
        if self.scene_mgr: self.scene_mgr.load_texture_to_selected(map_attr, filepath)

    def remove_texture_from_selected(self, map_attr: str) -> None:
        if self.scene_mgr: self.scene_mgr.remove_texture_from_selected(map_attr)

    def is_texture_in_use(self, path: str) -> bool:
        return self.scene_mgr.is_texture_in_use(path) if self.scene_mgr else False

    def get_scene_snapshot(self) -> str:
        return self.scene_mgr.get_scene_snapshot() if self.scene_mgr else ""

    def restore_snapshot(self, snapshot_str: str, current_aspect: float) -> None:
        if self.scene_mgr: self.scene_mgr.restore_snapshot(snapshot_str, current_aspect)

    def save_project(self, file_path: str, metadata: Dict[str, Any]) -> None:
        if self.scene_mgr: self.scene_mgr.save_project(file_path, metadata)

    def load_project(self, file_path: str, current_aspect: float) -> Dict[str, Any]:
        return self.scene_mgr.load_project(file_path, current_aspect) if self.scene_mgr else {}

    def export_scene_obj(self, export_dir: str) -> None:
        if self.scene_mgr: self.scene_mgr.export_scene_obj(export_dir)

    def add_empty_group(self) -> None:
        if self.entity_fac: self.entity_fac.add_empty_group()

    def spawn_primitive(self, name: str, is_2d: bool) -> None:
        if self.entity_fac: self.entity_fac.spawn_primitive(name, is_2d)

    def spawn_math_surface(self, formula: str, xmin: float, xmax: float, ymin: float, ymax: float, res: int) -> None:
        if self.entity_fac: self.entity_fac.spawn_math_surface(formula, xmin, xmax, ymin, ymax, res)

    def add_light(self, light_type: str, proxy_enabled: bool, global_light_on: bool) -> None:
        if self.entity_fac: self.entity_fac.add_light(light_type, proxy_enabled, global_light_on)

    def add_camera(self, proxy_enabled: bool) -> None:
        if self.entity_fac: self.entity_fac.add_camera(proxy_enabled)

    def spawn_model_from_path(self, path: str) -> None:
        if self.entity_fac: self.entity_fac.spawn_model_from_path(path)

    def check_gizmo_hover(self, mx: float, my: float, width: int, height: int, custom_tf=None, custom_view=None, custom_proj=None, is_hud=False) -> Optional[str]:
        return self.interaction_mgr.check_gizmo_hover(mx, my, width, height, custom_tf, custom_view, custom_proj, is_hud) if self.interaction_mgr else None

    def check_screen_axis_hover(self, mx: float, my: float, width: int, height: int) -> Optional[str]:
        return self.interaction_mgr.check_screen_axis_hover(mx, my, width, height) if self.interaction_mgr else None

    def check_hud_gizmo_hover(self, mx: float, my: float, width: int, height: int) -> Optional[str]:
        return self.interaction_mgr.check_hud_gizmo_hover(mx, my, width, height) if self.interaction_mgr else None

    def handle_hud_gizmo_drag(self, dx: float, dy: float, active_axis: str, width: int, height: int) -> None:
        if self.interaction_mgr: self.interaction_mgr.handle_hud_gizmo_drag(dx, dy, active_axis, width, height)

    def handle_gizmo_drag(self, dx: float, dy: float, active_axis: str, width: int, height: int) -> None:
        if self.interaction_mgr: self.interaction_mgr.handle_gizmo_drag(dx, dy, active_axis, width, height)

    def update_camera_movement(self, active_commands: List[str], dt: float) -> bool:
        return self.interaction_mgr.update_camera_movement(active_commands, dt) if self.interaction_mgr else False

    def orbit_camera(self, dx: float, dy: float) -> None:
        if self.interaction_mgr: self.interaction_mgr.orbit_camera(dx, dy)

    def pan_camera(self, dx: float, dy: float) -> None:
        if self.interaction_mgr: self.interaction_mgr.pan_camera(dx, dy)

    def zoom_camera(self, yoffset: float) -> None:
        if self.interaction_mgr: self.interaction_mgr.zoom_camera(yoffset)

    def snap_camera_to_axis(self, axis: str) -> None:
        if self.interaction_mgr: self.interaction_mgr.snap_camera_to_axis(axis)

    def get_screen_axis_labels_data(self, width: int, height: int) -> List[Dict[str, Any]]:
        return self.interaction_mgr.get_screen_axis_labels_data(width, height) if self.interaction_mgr else []

    def toggle_wireframe(self) -> None:
        if self.renderer: self.renderer.toggle_wireframe()

    def set_render_settings(self, wireframe: bool, mode: int, output: int, light: bool, tex: bool, vcolor: bool) -> None:
        if self.renderer: self.renderer.set_render_settings(wireframe, mode, output, light, tex, vcolor)

    def get_semantic_classes(self) -> dict:
        return self.scene_mgr.get_semantic_classes() if self.scene_mgr else {}

    def add_semantic_class(self, name: str) -> int:
        return self.scene_mgr.add_semantic_class(name) if self.scene_mgr else -1

    def update_semantic_class_color(self, class_id: int, color: list) -> None:
        if self.scene_mgr: self.scene_mgr.update_semantic_class_color(class_id, color)

    def remove_semantic_class(self, class_id: int) -> None:
        if self.scene_mgr: self.scene_mgr.remove_semantic_class(class_id)

    def get_animation_info(self) -> dict:
        return self.scene_mgr.get_animation_info() if self.scene_mgr else {}

    def set_active_keyframe(self, index: int) -> float:
        return self.scene_mgr.set_active_keyframe(index) if self.scene_mgr else 0.0

    def sync_gizmo_to_keyframe(self, current_time: float, is_hud_drag: bool = False) -> Tuple[bool, float]:
        if self.scene_mgr and hasattr(self.scene_mgr.animation, 'sync_gizmo_to_keyframe'):
            return self.scene_mgr.animation.sync_gizmo_to_keyframe(current_time, is_hud_drag)
        return False, current_time

    def update_keyframe_property(self, current_time: float, comp_name: str, prop: str, value: Any) -> tuple:
        return self.scene_mgr.update_keyframe_property(current_time, comp_name, prop, value) if self.scene_mgr else (False, False, 0.0)

    def update_keyframe_properties(self, current_time: float, comp_name: str, payload: Dict[str, Any]) -> tuple:
        return self.scene_mgr.update_keyframe_properties(current_time, comp_name, payload) if self.scene_mgr else (False, False, 0.0)

    def add_and_focus_keyframe(self, time: float) -> int:
        return self.scene_mgr.add_and_focus_keyframe(time) if self.scene_mgr else -1

    def mutate_keyframes(self, payload: dict) -> None:
        if self.scene_mgr and self.scene.selected_index >= 0:
            ent = self.scene.entities[self.scene.selected_index]
            from src.engine.scene.components.animation_cmp import AnimationComponent
            anim = ent.get_component(AnimationComponent)
            if anim:
                self.scene_mgr.animation.handle_animation_property(ent, anim, "MUTATE_KEYFRAMES", payload)

    def get_resolved_track_id(self, ent: Any) -> int:
        if not self.scene or ent not in self.scene.entities:
            return -1
            
        from src.engine.scene.components.semantic_cmp import SemanticComponent
        from src.engine.scene.components import MeshRenderer
        
        sem = ent.get_component(SemanticComponent)
        if sem and getattr(sem, 'track_id', -1) > 0: 
            return sem.track_id
            
        target_map = {}
        dense_id = 1
        
        for e in self.scene.entities:
            e_sem = e.get_component(SemanticComponent)
            e_mesh = e.get_component(MeshRenderer)
            if e_sem and e_mesh and getattr(e_mesh, 'visible', True) and not getattr(e_mesh, 'is_proxy', False):
                t = e
                p = e.parent
                while p:
                    p_sem = p.get_component(SemanticComponent)
                    if p_sem and getattr(p_sem, 'is_merged_instance', False):
                        t = p
                    p = p.parent
                
                if t not in target_map:
                    target_map[t] = dense_id
                    dense_id += 1
                    
        target = ent
        p = ent.parent
        while p:
            p_sem = p.get_component(SemanticComponent)
            if p_sem and getattr(p_sem, 'is_merged_instance', False):
                target = p
            p = p.parent
            
        return target_map.get(target, -1)