"""
Geometry Exporter.
Serializes scene geometry and materials into Wavefront OBJ and MTL formats.
"""

import shutil
import numpy as np
import glm
from pathlib import Path
from typing import Dict, List, Any, TextIO

from src.engine.scene.entity import Entity
from src.engine.scene.components import TransformComponent, MeshRenderer
from src.app.exceptions import ResourceError
from src.app.config import MTL_TOKENS

class OBJExporter:
    """
    Handles the extraction and formatting of scene data for external 3D applications.
    """

    @classmethod
    def export(cls, entities: List[Entity], export_dir_str: str) -> None:
        """Serializes the provided entities into the target directory."""
        export_dir = Path(export_dir_str)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        obj_path = export_dir / "models.obj"
        mtl_path = export_dir / "materials.mtl"
        tex_folder_name = "textures"
        tex_dir_full = export_dir / tex_folder_name

        try:
            with obj_path.open("w", encoding="utf-8") as f_obj, mtl_path.open("w", encoding="utf-8") as f_mtl:
                f_obj.write("mtllib materials.mtl\n")
                export_state = {"v_offset": 1, "mat_count": 0}

                for ent in entities:
                    cls._export_entity_to_obj(ent, f_obj, f_mtl, tex_dir_full, tex_folder_name, export_state)
        except Exception as e:
            raise ResourceError(f"Failed to export scene geometry to '{export_dir}'.\nReason: {e}")

    @classmethod
    def _export_entity_to_obj(
        cls,
        ent: Entity,
        f_obj: TextIO,
        f_mtl: TextIO,
        tex_dir_full: Path,
        tex_folder_name: str,
        state: Dict[str, int],
    ) -> None:
        """Recursively parses an entity and its children into OBJ/MTL formatting."""
        mesh = ent.get_component(MeshRenderer)
        tf = ent.get_component(TransformComponent)

        if mesh and mesh.visible and not getattr(mesh, "is_proxy", False) and mesh.geometry:
            mat_name = f"mat_{state['mat_count']}_{ent.name.replace(' ', '_')}"
            state["mat_count"] += 1
            mat = mesh.material

            # Write Material Properties
            f_mtl.write(f"newmtl {mat_name}\n")
            f_mtl.write(f"Ka {mat._ambient.x:.4f} {mat._ambient.y:.4f} {mat._ambient.z:.4f}\n")
            f_mtl.write(f"Kd {mat._diffuse.x:.4f} {mat._diffuse.y:.4f} {mat._diffuse.z:.4f}\n")
            f_mtl.write(f"Ks {mat._specular.x:.4f} {mat._specular.y:.4f} {mat._specular.z:.4f}\n")
            f_mtl.write(f"Ns {getattr(mat, 'shininess', 32.0):.4f}\n")

            # Package Textures
            tex_paths_dict = getattr(mat, "tex_paths", {})
            for map_attr, t_path_str in tex_paths_dict.items():
                if t_path_str:
                    t_path = Path(t_path_str)
                    if t_path.exists():
                        tex_dir_full.mkdir(exist_ok=True)
                        dest_tex = tex_dir_full / t_path.name
                        try:
                            if not dest_tex.exists() or not dest_tex.samefile(t_path):
                                shutil.copy2(t_path, dest_tex)
                            token = MTL_TOKENS.get(map_attr, "map_Kd")
                            f_mtl.write(f"{token} {tex_folder_name}/{t_path.name}\n")
                        except Exception as e:
                            raise ResourceError(f"Texture packaging failed during export: {e}")

            f_mtl.write("\n")
            
            # Write Geometry
            f_obj.write(f"o {ent.name.replace(' ', '_')}\n")
            f_obj.write(f"usemtl {mat_name}\n")

            geom = mesh.geometry
            if not hasattr(geom, "vertices") or not hasattr(geom, "vertex_size"):
                return

            v_size = geom.vertex_size
            verts = np.array(geom.vertices, dtype=np.float32).reshape(-1, v_size)
            num_v = len(verts)
            
            global_mat = tf.get_matrix() if tf else glm.mat4(1.0)
            norm_mat = glm.transpose(glm.inverse(glm.mat3(global_mat)))

            # Export World-Space Positions
            for i in range(num_v):
                pos = glm.vec3(verts[i][0], verts[i][1], verts[i][2])
                g_pos = glm.vec3(global_mat * glm.vec4(pos, 1.0))
                f_obj.write(f"v {g_pos.x:.6f} {g_pos.y:.6f} {g_pos.z:.6f}\n")

            # Export UVs
            for i in range(num_v):
                f_obj.write(f"vt {verts[i][6]:.6f} {verts[i][7]:.6f}\n")

            # Export World-Space Normals
            for i in range(num_v):
                norm = glm.vec3(verts[i][3], verts[i][4], verts[i][5])
                g_norm = glm.normalize(norm_mat * norm) if glm.length(norm) > 0 else glm.vec3(0, 1, 0)
                f_obj.write(f"vn {g_norm.x:.6f} {g_norm.y:.6f} {g_norm.z:.6f}\n")

            # Export Faces/Indices
            if getattr(geom, "indices", None) is not None and len(geom.indices) > 0:
                idx = geom.indices
                v_off = state["v_offset"]
                for i in range(0, len(idx), 3):
                    i1, i2, i3 = idx[i], idx[i + 1], idx[i + 2]
                    f_obj.write(
                        f"f {i1+v_off}/{i1+v_off}/{i1+v_off} "
                        f"{i2+v_off}/{i2+v_off}/{i2+v_off} "
                        f"{i3+v_off}/{i3+v_off}/{i3+v_off}\n"
                    )
            else:
                v_off = state["v_offset"]
                for i in range(num_v):
                    f_obj.write(f"p {i+v_off}\n")

            state["v_offset"] += num_v

        # Traverse Hierarchy
        for child in ent.children:
            cls._export_entity_to_obj(child, f_obj, f_mtl, tex_dir_full, tex_folder_name, state)