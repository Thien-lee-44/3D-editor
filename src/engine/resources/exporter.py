import os
import shutil
import numpy as np
import glm
from typing import Dict, List, Any

from src.engine.scene.entity import Entity
from src.engine.scene.components import TransformComponent, MeshRenderer
from src.app.exceptions import ResourceError
from src.app.config import MTL_TOKENS

class OBJExporter:
    """
    Handles the serialization of internal Entity hierarchy and geometry data
    into the standard Wavefront (.obj) and Material (.mtl) formats.
    """

    @classmethod
    def export(cls, entities: List[Entity], export_dir: str) -> None:
        obj_path = os.path.join(export_dir, "models.obj")
        mtl_path = os.path.join(export_dir, "materials.mtl")
        tex_folder_name = "textures" 
        tex_dir_full = os.path.join(export_dir, tex_folder_name)
        
        with open(obj_path, 'w', encoding='utf-8') as f_obj, open(mtl_path, 'w', encoding='utf-8') as f_mtl:
            f_obj.write("mtllib materials.mtl\n")
            export_state = {'v_offset': 1, 'mat_count': 0}
            
            try:
                for ent in entities:
                    cls._export_entity_to_obj(ent, f_obj, f_mtl, tex_dir_full, tex_folder_name, export_state)
            except Exception as e:
                raise ResourceError(f"Failed to export scene geometry to '{export_dir}'.\nReason: {e}")

    @classmethod
    def _export_entity_to_obj(cls, ent: Entity, f_obj: Any, f_mtl: Any, tex_dir_full: str, tex_folder_name: str, state: Dict[str, int]) -> None:
        mesh = ent.get_component(MeshRenderer)
        tf = ent.get_component(TransformComponent)
        
        if mesh and mesh.visible and not getattr(mesh, 'is_proxy', False) and mesh.geometry:
            mat_name = f"mat_{state['mat_count']}_{ent.name.replace(' ', '_')}"
            state['mat_count'] += 1
            mat = mesh.material
            
            f_mtl.write(f"newmtl {mat_name}\n")
            f_mtl.write(f"Ka {mat._ambient.x:.4f} {mat._ambient.y:.4f} {mat._ambient.z:.4f}\n")
            f_mtl.write(f"Kd {mat._diffuse.x:.4f} {mat._diffuse.y:.4f} {mat._diffuse.z:.4f}\n")
            f_mtl.write(f"Ks {mat._specular.x:.4f} {mat._specular.y:.4f} {mat._specular.z:.4f}\n")
            f_mtl.write(f"Ns {getattr(mat, 'shininess', 32.0):.4f}\n")
            
            tex_paths_dict = getattr(mat, 'tex_paths', {})
            for map_attr, t_path in tex_paths_dict.items():
                if t_path and os.path.exists(t_path):
                    tex_name = os.path.basename(t_path)
                    if not os.path.exists(tex_dir_full): 
                        os.makedirs(tex_dir_full)
                    dest_tex = os.path.join(tex_dir_full, tex_name)
                    try:
                        if not os.path.exists(dest_tex) or not os.path.samefile(t_path, dest_tex): 
                            shutil.copy2(t_path, dest_tex)
                        token = MTL_TOKENS.get(map_attr, "map_Kd")
                        f_mtl.write(f"{token} {tex_folder_name}/{tex_name}\n")
                    except Exception as e:
                        raise ResourceError(f"Texture packaging failed during export: {e}")
            f_mtl.write("\n")
            f_obj.write(f"o {ent.name.replace(' ', '_')}\n")
            f_obj.write(f"usemtl {mat_name}\n")
            
            geom = mesh.geometry
            if not hasattr(geom, 'vertices') or not hasattr(geom, 'vertex_size'): 
                return
                
            v_size = geom.vertex_size
            verts = np.array(geom.vertices, dtype=np.float32).reshape(-1, v_size)
            num_v = len(verts)
            global_mat = tf.get_matrix() if tf else glm.mat4(1.0)
            norm_mat = glm.transpose(glm.inverse(glm.mat3(global_mat)))
            
            for i in range(num_v):
                pos = glm.vec3(verts[i][0], verts[i][1], verts[i][2])
                g_pos = glm.vec3(global_mat * glm.vec4(pos, 1.0))
                f_obj.write(f"v {g_pos.x:.6f} {g_pos.y:.6f} {g_pos.z:.6f}\n")
                
            for i in range(num_v): 
                f_obj.write(f"vt {verts[i][6]:.6f} {verts[i][7]:.6f}\n")
                
            for i in range(num_v):
                norm = glm.vec3(verts[i][3], verts[i][4], verts[i][5])
                g_norm = glm.normalize(norm_mat * norm) if glm.length(norm) > 0 else glm.vec3(0, 1, 0)
                f_obj.write(f"vn {g_norm.x:.6f} {g_norm.y:.6f} {g_norm.z:.6f}\n")
                
            if getattr(geom, 'indices', None) is not None and len(geom.indices) > 0:
                idx = geom.indices
                v_off = state['v_offset']
                for i in range(0, len(idx), 3):
                    i1, i2, i3 = idx[i], idx[i+1], idx[i+2]
                    f_obj.write(f"f {i1+v_off}/{i1+v_off}/{i1+v_off} {i2+v_off}/{i2+v_off}/{i2+v_off} {i3+v_off}/{i3+v_off}/{i3+v_off}\n")
            else:
                v_off = state['v_offset']
                for i in range(num_v): 
                    f_obj.write(f"p {i+v_off}\n")
                    
            state['v_offset'] += num_v
            
        for child in ent.children:
            cls._export_entity_to_obj(child, f_obj, f_mtl, tex_dir_full, tex_folder_name, state)