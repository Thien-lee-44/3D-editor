"""
3D Geometry Parser.
Supports Wavefront (.obj) and Stanford (.ply) formats.
Includes advanced sub-mesh isolation and pivot baking for local coordinate correction.
"""

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
    Parser for 3D geometry formats.
    Loads and processes 3D model data, handling vertices, normals, UVs, and materials.
    """
    
    @staticmethod
    def load(filepath: str, normalize: Optional[bool] = None) -> List[BufferObject]:
        """
        Entry point to load a 3D model. Routes to specific format parsers.
        Automatically normalizes models loaded from outside the core asset directory.
        """
        if normalize is None:
            abs_mod_dir = str(MODELS_DIR.resolve()).replace('\\', '/')
            abs_file_path = str(Path(filepath).resolve()).replace('\\', '/')
            normalize = abs_mod_dir not in abs_file_path
            
        if filepath.lower().endswith('.obj'): 
            return ModelLoader._load_obj_custom(filepath, normalize)
        elif filepath.lower().endswith('.ply'): 
            return ModelLoader._load_ply_custom(filepath, normalize)
            
        raise ResourceError(f"Unsupported geometry format or file extension: '{filepath}'")
    
    @staticmethod
    def _normalize_vertices(vertex_data: np.ndarray, v_size: int = 8) -> np.ndarray:
        """
        Centers the mesh at the origin and scales it uniformly to fit within a unit bounding box.
        """
        pos = vertex_data.reshape(-1, v_size)[:, :3]
        min_b, max_b = pos.min(axis=0), pos.max(axis=0)
        max_dim = max(np.max(max_b - min_b), 1e-4)
        
        pos -= (min_b + max_b) * 0.5 
        pos *= (2.0 / max_dim)       
        return vertex_data

    @staticmethod
    def _load_obj_custom(filepath: str, normalize: bool) -> List[BufferObject]:
        """
        Parses Wavefront OBJ files. 
        Supports multi-object and multi-material extraction into separate BufferObjects.
        """
        parsed_mtl = ModelLoader._parse_materials(filepath)
        v_raw, vt_raw, vn_raw, vc_raw = [], [], [], []
        
        submeshes = {} 
        default_obj_name = os.path.splitext(os.path.basename(filepath))[0]
        curr_obj = default_obj_name
        curr_mat = 'default'
        
        unique_verts = {}
        vi_list, vti_list, vni_list = [], [], []
        idx_count = 0

        # Optimization: Local references to append methods for faster iteration
        v_app, vt_app, vn_app, vc_app = v_raw.append, vt_raw.append, vn_raw.append, vc_raw.append
        vi_app, vti_app, vni_app = vi_list.append, vti_list.append, vni_list.append

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line or line[0] == '#': 
                        continue
                        
                    parts = line.split()
                    if not parts: 
                        continue
                    
                    cmd = parts[0]
                    if cmd == 'v':
                        v_app((float(parts[1]), float(parts[2]), float(parts[3])))
                        if len(parts) >= 7:
                            vc_app((float(parts[4]), float(parts[5]), float(parts[6])))
                    
                    elif cmd in ('f', 'l', 'p'):
                        key = (curr_obj, curr_mat)
                        if key not in submeshes: 
                            submeshes[key] = {'faces': [], 'lines': [], 'points': []}
                            
                        parsed_elements = []
                        lv, lvt, lvn = len(v_raw), len(vt_raw), len(vn_raw)
                        
                        for tv in parts[1:]:
                            if tv not in unique_verts:
                                unique_verts[tv] = idx_count
                                p = tv.split('/')
                                
                                # Parse Vertex Index
                                s = p[0]
                                if s:
                                    vi_app(int(s) - 1 if int(s) > 0 else lv + int(s))
                                else:
                                    vi_app(-1)
                                    
                                # Parse UV Index
                                s = p[1] if len(p) > 1 else ''
                                if s:
                                    vti_app(int(s) - 1 if int(s) > 0 else lvt + int(s))
                                else:
                                    vti_app(-1)
                                    
                                # Parse Normal Index
                                s = p[2] if len(p) > 2 else ''
                                if s:
                                    vni_app(int(s) - 1 if int(s) > 0 else lvn + int(s))
                                else:
                                    vni_app(-1)
                                
                                idx_count += 1
                                
                            parsed_elements.append(unique_verts[tv])
                            
                        # Triangulate and assign primitive types
                        if cmd == 'f':
                            for i in range(1, len(parsed_elements) - 1):
                                submeshes[key]['faces'].extend([parsed_elements[0], parsed_elements[i], parsed_elements[i+1]])
                        elif cmd == 'l':
                            for i in range(len(parsed_elements) - 1):
                                submeshes[key]['lines'].extend([parsed_elements[i], parsed_elements[i+1]])
                        elif cmd == 'p':
                            submeshes[key]['points'].extend(parsed_elements)
                                
                    elif cmd == 'vt': 
                        vt_app((float(parts[1]), float(parts[2]))) 
                    elif cmd == 'vn': 
                        vn_app((float(parts[1]), float(parts[2]), float(parts[3])))
                    elif cmd == 'usemtl':
                        curr_mat = parts[1].strip(' "\'') if len(parts) > 1 else 'default'
                    elif cmd in ('o', 'g'):
                        curr_obj = parts[1].strip(' "\'') if len(parts) > 1 else f'object_{len(submeshes)}'
                        
        except Exception as e:
            raise ResourceError(f"Failed to read or parse OBJ file '{filepath}'.\nReason: {e}")
        
        if not vi_list: 
            return []
        
        # Zero-padding for unassigned attributes (used when index is -1)
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

        # Fallback: Calculate flat normals if missing
        if len(vn_raw) == 1:
            all_idx = [idx for d in submeshes.values() for idx in d['faces']]
            if all_idx:
                faces = np.array(all_idx, dtype=np.uint32).reshape(-1, 3)
                valid_faces = faces[(faces < len(positions)).all(axis=1)]
                
                if len(valid_faces) > 0:
                    p0 = positions[valid_faces[:, 0]]
                    p1 = positions[valid_faces[:, 1]]
                    p2 = positions[valid_faces[:, 2]]
                    
                    cross = np.cross(p1 - p0, p2 - p0)
                    np.add.at(normals, valid_faces[:, 0], cross)
                    np.add.at(normals, valid_faces[:, 1], cross)
                    np.add.at(normals, valid_faces[:, 2], cross)
                    
                    norms_len = np.linalg.norm(normals, axis=1, keepdims=True)
                    np.divide(normals, norms_len, out=normals, where=norms_len!=0)

        # 11 = pos(3) + norm(3) + uv(2) + col(3); 8 = pos(3) + norm(3) + uv(2)
        v_size = 11 if vc_raw else 8
        if v_size == 11:
            vertex_data = np.hstack((positions, normals, uvs, colors)).astype(np.float32)
        else:
            vertex_data = np.hstack((positions, normals, uvs)).astype(np.float32)
            
        if normalize: 
            vertex_data = ModelLoader._normalize_vertices(vertex_data, v_size)
        
        result = []
        vd_reshaped = vertex_data.reshape(-1, v_size)
        
        obj_part_count = {}
        for (o_name, m_name), d in submeshes.items():
            active_types = sum(1 for k in ['faces', 'lines', 'points'] if d[k])
            obj_part_count[o_name] = obj_part_count.get(o_name, 0) + active_types

        # Build final BufferObjects for each isolated sub-mesh
        for (obj_name, mat_name), d in submeshes.items():
            idx_arrays = []
            if d['faces']: idx_arrays.append(d['faces'])
            if d['lines']: idx_arrays.append(d['lines'])
            if d['points']: idx_arrays.append(d['points'])
            
            if not idx_arrays: 
                continue
            
            all_idx = np.concatenate(idx_arrays)
            used_indices = np.unique(all_idx)
            
            local_vd = vd_reshaped[used_indices].copy()
            sub_pos = local_vd[:, :3]
            min_b, max_b = sub_pos.min(axis=0), sub_pos.max(axis=0)
            center = (min_b + max_b) / 2.0
            
            # Bake local pivot offset
            local_vd[:, :3] -= center
            
            remap_arr = np.zeros(used_indices.max() + 1, dtype=np.uint32)
            remap_arr[used_indices] = np.arange(len(used_indices), dtype=np.uint32)
            
            local_faces = remap_arr[d['faces']] if d['faces'] else None
            local_lines = remap_arr[d['lines']] if d['lines'] else None
            local_points = remap_arr[d['points']] if d['points'] else None

            # Fetch Material
            mat_dict = parsed_mtl.get(mat_name, {
                'ambient': [1]*3, 'diffuse': [0.9]*3, 'specular': [0.2]*3, 
                'shininess': 32.0, 'texture': ""
            }).copy()
            
            if mat_dict.get('texture'): 
                mat_dict['ambient'], mat_dict['diffuse'] = [1]*3, [1]*3
            
            is_multi_part = obj_part_count[obj_name] > 1
            base_name = f"{obj_name}_{mat_name}" if is_multi_part else obj_name
            active_types = sum(1 for k in ['faces', 'lines', 'points'] if d[k])
            
            def create_buffer(geom_data: np.ndarray, indices: Optional[np.ndarray], mode: int, suffix: str) -> None:
                name = base_name + suffix if active_types > 1 else base_name
                geom = BufferObject(geom_data.flatten(), indices, v_size, render_mode=mode)
                geom.name = name
                geom.group_name = obj_name  
                geom.filepath = filepath
                geom.materials = {'default_active': mat_dict}
                geom.pivot_offset = center.tolist()
                result.append(geom)

            if local_faces is not None: create_buffer(local_vd, local_faces, GL_TRIANGLES, "_Faces")
            if local_lines is not None: create_buffer(local_vd, local_lines, GL_LINES, "_Lines")
            if local_points is not None: create_buffer(local_vd, local_points, GL_POINTS, "_Points")

        return result

    @staticmethod
    def _parse_materials(obj_filepath: str) -> Dict[str, Any]:
        """
        Extracts material properties and texture mappings from the associated MTL file.
        Uses string-based path resolution to guarantee compatibility with external asset directories.
        """
        mats = {}
        mtl_path = None
        obj_dir = os.path.dirname(obj_filepath)
        
        try:
            with open(obj_filepath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i > 1000: break # Prevent infinite scanning if file is broken
                    if line.startswith('mtllib '):
                        parts = line.split(None, 1)
                        if len(parts) > 1: 
                            mtl_path = os.path.join(obj_dir, parts[1].strip(' "\'').replace('\\', '/'))
                        break
        except Exception: 
            pass

        # Fallback mechanism if exact match is not found
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
                if os.path.exists(fb): 
                    tex = fb
            return tex

        curr_mat = None
        try:
            with open(mtl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(None, 1)
                    if len(parts) < 2 or line[0] == '#': 
                        continue
                        
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
                        if cmd in ('ka', 'kd', 'ks'): 
                            target = {'ka':'ambient', 'kd':'diffuse', 'ks':'specular'}[cmd]
                            mats[curr_mat][target] = [float(x) for x in val.split()[:3]]
                        elif cmd == 'ke': mats[curr_mat]['emission'] = [float(x) for x in val.split()[:3]]
                        elif cmd == 'ns': mats[curr_mat]['shininess'] = float(val.split()[0])
                        elif cmd == 'd': mats[curr_mat]['opacity'] = float(val.split()[0])
                        elif cmd == 'tr': mats[curr_mat]['opacity'] = 1.0 - float(val.split()[0])
                        elif cmd == 'ni': mats[curr_mat]['ior'] = float(val.split()[0])
                        elif cmd == 'illum': mats[curr_mat]['illum'] = int(val.split()[0])
                        
                        # Texture Maps
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
        """
        Parses Stanford PLY models.
        Extracts positions, colors, normals, and indices from binary/ascii datasets.
        """
        try:
            plydata = PlyData.read(filepath)
        except Exception as e:
            raise ResourceError(f"Failed to decode PLY buffer from '{filepath}'.\nReason: {e}")

        v_elements = plydata['vertex'].data
        names = v_elements.dtype.names
        num_verts = len(v_elements)
        positions = np.vstack((v_elements['x'], v_elements['y'], v_elements['z'])).T

        # Process Colors
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

        # Process Indices
        indices = []
        if 'face' in plydata:
            f_data = plydata['face'].data
            prop = 'vertex_indices' if 'vertex_indices' in f_data.dtype.names else 'vertex_index'
            if prop in f_data.dtype.names:
                for face in f_data[prop]:
                    for j in range(1, len(face) - 1):
                        indices.extend([face[0], face[j], face[j+1]])

        topology = GL_TRIANGLES if len(indices) > 0 else GL_POINTS

        # Process Normals
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
            
        min_b, max_b = vertex_data[:, :3].min(axis=0), vertex_data[:, :3].max(axis=0)
        center = (min_b + max_b) / 2.0
        vertex_data[:, :3] -= center

        idx_arr = np.array(indices, dtype=np.uint32) if indices else None
        
        geom = BufferObject(vertex_data.flatten(), idx_arr, v_size, render_mode=topology)
        geom.name = os.path.basename(filepath)
        geom.group_name = geom.name
        geom.filepath = filepath
        geom.materials = {}
        geom.pivot_offset = center.tolist()
        
        return [geom]