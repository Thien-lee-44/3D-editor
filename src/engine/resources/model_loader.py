import os
import numpy as np
from plyfile import PlyData
from typing import List, Dict, Optional, Any
from pathlib import Path
from OpenGL.GL import GL_TRIANGLES, GL_LINES, GL_POINTS

from src.engine.graphics.buffer_objects import BufferObject
from src.app.exceptions import ResourceError

from src.app.config import MODELS_DIR

class ModelLoader:
    """
    Parser for 3D geometry formats. Currently supports Wavefront (.obj) and Stanford (.ply) formats.
    """
    
    @staticmethod
    def load(filepath: str, normalize: Optional[bool] = None) -> List[BufferObject]:
        if normalize is None:
            abs_mod_dir = str(MODELS_DIR.resolve()).replace('\\', '/')
            abs_file_path = str(Path(filepath).resolve()).replace('\\', '/')
            normalize = abs_mod_dir not in abs_file_path
            
        if filepath.endswith('.obj'): 
            return ModelLoader._load_obj_custom(filepath, normalize)
        elif filepath.endswith('.ply'): 
            return ModelLoader._load_ply_custom(filepath, normalize)
            
        raise ResourceError(f"Unsupported geometry format or file extension: '{filepath}'")
    
    @staticmethod
    def _normalize_vertices(vertex_data: np.ndarray, v_size: int = 8) -> np.ndarray:
        pos = vertex_data.reshape(-1, v_size)[:, :3]
        min_b, max_b = pos.min(axis=0), pos.max(axis=0)
        max_dim = max(np.max(max_b - min_b), 1e-4)
        pos -= (min_b + max_b) * 0.5 
        pos *= (2.0 / max_dim)       
        return vertex_data

    @staticmethod
    def _load_obj_custom(filepath: str, normalize: bool) -> List[BufferObject]:
        parsed_mtl = ModelLoader._parse_materials(filepath)
        v_raw, vt_raw, vn_raw, vc_raw = [], [], [], []
        submeshes = {}
        curr_mat, curr_obj = 'default', os.path.basename(filepath)
        unique_verts = {}
        
        vi_list, vti_list, vni_list = [], [], []
        idx_count = 0

        v_app, vt_app, vn_app, vc_app = v_raw.append, vt_raw.append, vn_raw.append, vc_raw.append
        vi_app, vti_app, vni_app = vi_list.append, vti_list.append, vni_list.append

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line or line[0] == '#': continue
                    parts = line.split()
                    if not parts: continue
                    
                    cmd = parts[0]
                    if cmd == 'v':
                        v_app((float(parts[1]), float(parts[2]), float(parts[3])))
                        if len(parts) >= 7:
                            vc_app((float(parts[4]), float(parts[5]), float(parts[6])))
                    
                    elif cmd in ('f', 'l', 'p'):
                        if curr_mat not in submeshes: 
                            submeshes[curr_mat] = {'faces': [], 'lines': [], 'points': [], 'name': f"{curr_obj}_{curr_mat}"}
                            
                        parsed_elements = []
                        lv, lvt, lvn = len(v_raw), len(vt_raw), len(vn_raw)
                        
                        for tv in parts[1:]:
                            if tv not in unique_verts:
                                unique_verts[tv] = idx_count
                                p = tv.split('/')
                                
                                s = p[0]
                                vi_app(int(s) - 1 if int(s) > 0 else lv + int(s)) if s else vi_app(-1)
                                    
                                s = p[1] if len(p) > 1 else ''
                                vti_app(int(s) - 1 if int(s) > 0 else lvt + int(s)) if s else vti_app(-1)
                                    
                                s = p[2] if len(p) > 2 else ''
                                vni_app(int(s) - 1 if int(s) > 0 else lvn + int(s)) if s else vni_app(-1)
                                
                                idx_count += 1
                            parsed_elements.append(unique_verts[tv])
                            
                        if cmd == 'f':
                            for i in range(1, len(parsed_elements) - 1):
                                submeshes[curr_mat]['faces'].extend([parsed_elements[0], parsed_elements[i], parsed_elements[i+1]])
                        elif cmd == 'l':
                            for i in range(len(parsed_elements) - 1):
                                submeshes[curr_mat]['lines'].extend([parsed_elements[i], parsed_elements[i+1]])
                        elif cmd == 'p':
                            submeshes[curr_mat]['points'].extend(parsed_elements)
                                
                    elif cmd == 'vt': 
                        vt_app((float(parts[1]), float(parts[2]))) 
                    elif cmd == 'vn': 
                        vn_app((float(parts[1]), float(parts[2]), float(parts[3])))
                    elif cmd == 'usemtl':
                        curr_mat = parts[1].strip(' "\'') if len(parts) > 1 else 'default'
                        if curr_mat not in submeshes: 
                            submeshes[curr_mat] = {'faces': [], 'lines': [], 'points': [], 'name': f"{curr_obj}_{curr_mat}"}
                    elif cmd in ('o', 'g'):
                        curr_obj = parts[1].strip(' "\'') if len(parts) > 1 else 'object'
        except Exception as e:
            raise ResourceError(f"Failed to read or parse OBJ file '{filepath}'.\nReason: {e}")
        
        if not vi_list: return []
        
        v_raw.append((0.0, 0.0, 0.0))
        vt_raw.append((0.0, 0.0))
        vn_raw.append((0.0, 0.0, 0.0))
        if vc_raw:
            vc_raw.append((1.0, 1.0, 1.0))
        
        v_arr = np.array(v_raw, dtype=np.float32)
        vt_arr = np.array(vt_raw, dtype=np.float32)
        vn_arr = np.array(vn_raw, dtype=np.float32)

        positions = v_arr[vi_list]
        uvs = vt_arr[vti_list]
        normals = vn_arr[vni_list]
        
        if vc_raw:
            vc_arr = np.array(vc_raw, dtype=np.float32)
            colors = vc_arr[vi_list]
        else:
            colors = np.ones((len(positions), 3), dtype=np.float32)

        if len(vn_raw) == 1:
            all_idx = [idx for d in submeshes.values() for idx in d['faces']]
            if all_idx:
                faces = np.array(all_idx, dtype=np.uint32).reshape(-1, 3)
                valid_faces = faces[(faces < len(positions)).all(axis=1)]
                
                if len(valid_faces) > 0:
                    p0, p1, p2 = positions[valid_faces[:, 0]], positions[valid_faces[:, 1]], positions[valid_faces[:, 2]]
                    cross = np.cross(p1 - p0, p2 - p0)
                    np.add.at(normals, valid_faces[:, 0], cross)
                    np.add.at(normals, valid_faces[:, 1], cross)
                    np.add.at(normals, valid_faces[:, 2], cross)
                    norms_len = np.linalg.norm(normals, axis=1, keepdims=True)
                    np.divide(normals, norms_len, out=normals, where=norms_len!=0)

        v_size = 11 if vc_raw else 8
        
        if v_size == 11:
            vertex_data = np.hstack((positions, normals, uvs, colors)).astype(np.float32)
        else:
            vertex_data = np.hstack((positions, normals, uvs)).astype(np.float32)
            
        if normalize: 
            vertex_data = ModelLoader._normalize_vertices(vertex_data, v_size)
        
        result = []
        vd_reshaped = vertex_data.reshape(-1, v_size)
        
        for mat_name, d in submeshes.items():
            mat_dict = parsed_mtl.get(mat_name, {'ambient': [1]*3, 'diffuse': [0.9]*3, 'specular': [0.2]*3, 'shininess': 32.0, 'texture': ""}).copy()
            if mat_dict.get('texture'): 
                mat_dict['ambient'], mat_dict['diffuse'] = [1]*3, [1]*3
            
            active_types = sum(1 for k in ['faces', 'lines', 'points'] if d[k])
            has_rendered = False
            if d['faces']:
                name = d['name'] + ("_Faces" if active_types > 1 else "")
                geom = BufferObject(vertex_data.flatten(), np.array(d['faces'], dtype=np.uint32), v_size, render_mode=GL_TRIANGLES)
                geom.name = name
                geom.filepath = filepath
                geom.materials = {'default_active': mat_dict}
                result.append(geom)
                has_rendered = True

            if d['lines']:
                name = d['name'] + ("_Lines" if active_types > 1 else "")
                geom = BufferObject(vertex_data.flatten(), np.array(d['lines'], dtype=np.uint32), v_size, render_mode=GL_LINES)
                geom.name = name
                geom.filepath = filepath
                geom.materials = {'default_active': mat_dict}
                result.append(geom)
                has_rendered = True

            if d['points']:
                name = d['name'] + ("_Points" if active_types > 1 else "")
                pt_data = vd_reshaped[d['points']]
                geom = BufferObject(pt_data.flatten(), None, v_size, render_mode=GL_POINTS)
                geom.name = name
                geom.filepath = filepath
                geom.materials = {'default_active': mat_dict}
                result.append(geom)
                has_rendered = True
                
            if not has_rendered:
                name = d['name'] + "_RawPoints"
                geom = BufferObject(vertex_data.flatten(), None, v_size, render_mode=GL_POINTS)
                geom.name = name
                geom.filepath = filepath
                geom.materials = {'default_active': mat_dict}
                result.append(geom)

        return result

    @staticmethod
    def _parse_materials(obj_filepath: str) -> Dict[str, Any]:
        mats, mtl_path, obj_dir = {}, None, os.path.dirname(obj_filepath)
        try:
            with open(obj_filepath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i > 1000: break
                    if line.startswith('mtllib '):
                        parts = line.split(None, 1)
                        if len(parts) > 1: mtl_path = os.path.join(obj_dir, parts[1].strip(' "\'').replace('\\', '/'))
                        break
        except Exception: 
            pass

        if not mtl_path or not os.path.exists(mtl_path):
            candidates = [f for f in os.listdir(obj_dir) if f.lower().endswith('.mtl')]
            if candidates:
                base = os.path.splitext(os.path.basename(obj_filepath))[0].lower()
                mtl_name = os.path.basename(mtl_path or '').lower().replace('_', ' ')
                best_match = next((f for f in candidates if f.lower().replace('_', ' ') == mtl_name), None)
                best_match = best_match or next((f for f in candidates if base in f.lower()), candidates[0])
                mtl_path = os.path.join(obj_dir, best_match)

        if not mtl_path or not os.path.exists(mtl_path): 
            return mats
        
        def resolve_tex(val: str) -> str:
            tex = val.strip(' "\'').replace('\\', '/')
            if not os.path.isabs(tex): 
                tex = os.path.normpath(os.path.join(obj_dir, tex))
            if not os.path.exists(tex):
                fb = os.path.join(obj_dir, os.path.basename(tex))
                if os.path.exists(fb): tex = fb
            return tex

        curr_mat = None
        try:
            with open(mtl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(None, 1)
                    if len(parts) < 2 or line[0] == '#': continue
                    cmd, val = parts[0].lower(), parts[1]
                    
                    if cmd == 'newmtl':
                        curr_mat = val.strip(' "\'')
                        mats[curr_mat] = {
                            'ambient': [1]*3, 'diffuse': [0.9]*3, 'specular': [0.2]*3, 'emission': [0]*3,
                            'shininess': 32.0, 'opacity': 1.0, 'ior': 1.0, 'illum': 2,
                            'map_diffuse': "", 'map_specular': "", 'map_bump': "",
                            'map_ambient': "", 'map_emission': "", 'map_shininess': "", 'map_opacity': "",
                            'map_reflection': ""
                        }
                    elif curr_mat:
                        if cmd in ('ka', 'kd', 'ks'): mats[curr_mat][{'ka':'ambient', 'kd':'diffuse', 'ks':'specular'}[cmd]] = [float(x) for x in val.split()[:3]]
                        elif cmd == 'ke': mats[curr_mat]['emission'] = [float(x) for x in val.split()[:3]]
                        elif cmd == 'ns': mats[curr_mat]['shininess'] = float(val.split()[0])
                        elif cmd == 'd': mats[curr_mat]['opacity'] = float(val.split()[0])
                        elif cmd == 'tr': mats[curr_mat]['opacity'] = 1.0 - float(val.split()[0])
                        elif cmd == 'ni': mats[curr_mat]['ior'] = float(val.split()[0])
                        elif cmd == 'illum': mats[curr_mat]['illum'] = int(val.split()[0])
                        elif cmd == 'map_kd': mats[curr_mat]['map_diffuse'] = resolve_tex(val)
                        elif cmd == 'map_ks': mats[curr_mat]['map_specular'] = resolve_tex(val)
                        elif cmd in ('map_bump', 'bump'): mats[curr_mat]['map_bump'] = resolve_tex(val)
                        elif cmd == 'map_ka': mats[curr_mat]['map_ambient'] = resolve_tex(val)
                        elif cmd == 'map_ke': mats[curr_mat]['map_emission'] = resolve_tex(val)
                        elif cmd == 'map_ns': mats[curr_mat]['map_shininess'] = resolve_tex(val)
                        elif cmd in ('map_d', 'map_opacity'): mats[curr_mat]['map_opacity'] = resolve_tex(val)
                        elif cmd == 'refl': mats[curr_mat]['map_reflection'] = resolve_tex(val)
        except Exception: 
            pass
            
        return mats

    @staticmethod
    def _load_ply_custom(filepath: str, normalize: bool) -> List[BufferObject]:
        try:
            plydata = PlyData.read(filepath)
        except Exception as e:
            raise ResourceError(f"Failed to decode PLY buffer from '{filepath}'.\nReason: {e}")

        v_elements = plydata['vertex'].data
        names = v_elements.dtype.names
        num_verts = len(v_elements)
        positions = np.vstack((v_elements['x'], v_elements['y'], v_elements['z'])).T

        colors = np.ones((num_verts, 3), dtype=np.float32)
        r_name = next((n for n in names if n in ['red', 'r', 'diffuse_red']), None)
        g_name = next((n for n in names if n in ['green', 'g', 'diffuse_green']), None)
        b_name = next((n for n in names if n in ['blue', 'b', 'diffuse_blue']), None)
        
        has_color = False
        if r_name and g_name and b_name:
            has_color = True
            colors = np.vstack((v_elements[r_name], v_elements[g_name], v_elements[b_name])).T
            if v_elements[r_name].dtype == np.uint8:
                colors = colors / 255.0

        indices = []
        if 'face' in plydata:
            f_data = plydata['face'].data
            prop = 'vertex_indices' if 'vertex_indices' in f_data.dtype.names else 'vertex_index'
            if prop in f_data.dtype.names:
                for face in f_data[prop]:
                    for j in range(1, len(face) - 1):
                        indices.extend([face[0], face[j], face[j+1]])

        topology = GL_TRIANGLES if len(indices) > 0 else GL_POINTS

        nx_name = next((n for n in names if n in ['nx', 'normal_x']), None)
        ny_name = next((n for n in names if n in ['ny', 'normal_y']), None)
        nz_name = next((n for n in names if n in ['nz', 'normal_z']), None)

        if nx_name and ny_name and nz_name:
            normals = np.vstack((v_elements[nx_name], v_elements[ny_name], v_elements[nz_name])).T
        else:
            normals = np.zeros((num_verts, 3), dtype=np.float32)
            if indices:
                faces = np.array(indices, dtype=np.uint32).reshape(-1, 3)
                p0 = positions[faces[:, 0]]
                p1 = positions[faces[:, 1]]
                p2 = positions[faces[:, 2]]
                
                cross = np.cross(p1 - p0, p2 - p0)
                np.add.at(normals, faces[:, 0], cross)
                np.add.at(normals, faces[:, 1], cross)
                np.add.at(normals, faces[:, 2], cross)
                
                norms_len = np.linalg.norm(normals, axis=1, keepdims=True)
                np.divide(normals, norms_len, out=normals, where=norms_len!=0)
            else:
                normals[:, 1] = 1.0 

        uvs = np.zeros((num_verts, 2), dtype=np.float32)
        
        v_size = 11 if has_color else 8
        if has_color:
            vertex_data = np.hstack((positions, normals, uvs, colors)).astype(np.float32)
        else:
            vertex_data = np.hstack((positions, normals, uvs)).astype(np.float32)
        
        if normalize: 
            vertex_data = ModelLoader._normalize_vertices(vertex_data, v_size)

        idx_arr = np.array(indices, dtype=np.uint32) if indices else None
        
        geom = BufferObject(vertex_data.flatten(), idx_arr, v_size, render_mode=topology)
        geom.name = os.path.basename(filepath)
        geom.filepath = filepath
        geom.materials = {}
        
        return [geom]