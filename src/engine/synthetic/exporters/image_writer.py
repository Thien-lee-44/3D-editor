import numpy as np
from PIL import Image
from typing import Optional

class ImageWriter:
    @staticmethod
    def save_rgb(filepath: str, pixel_data: bytes, width: int, height: int) -> None:
        arr = np.frombuffer(pixel_data, dtype=np.uint8).reshape((height, width, 3))
        img = Image.fromarray(arr, 'RGB')
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img.save(filepath, format="JPEG", quality=100)

    @staticmethod
    def save_mask(filepath: str, pixel_data: bytes, width: int, height: int) -> None:
        arr = np.frombuffer(pixel_data, dtype=np.uint8).reshape((height, width, 3))
        img = Image.fromarray(arr, 'RGB')
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img.save(filepath, format="PNG")

    @staticmethod
    def save_depth(
        filepath: str,
        depth_data: np.ndarray,
        width: int,
        height: int,
        near: Optional[float] = None,
        far: Optional[float] = None,
    ) -> None:
        depth_map = np.array(depth_data, dtype=np.float32).reshape((height, width))
        valid_mask = np.isfinite(depth_map) & (depth_map > 0.0)

        out_arr = np.zeros((height, width), dtype=np.uint8)

        if np.any(valid_mask):
            if near is not None and far is not None and far > near:
                d_min = float(near)
                d_max = float(far)
            else:
                valid_values = depth_map[valid_mask]
                d_min = float(np.min(valid_values))
                d_max = float(np.max(valid_values))

            if d_max > d_min:
                clamped = np.clip(depth_map, d_min, d_max)
                normalized = (clamped - d_min) / (d_max - d_min)
                out_arr[valid_mask] = (normalized[valid_mask] * 255.0).astype(np.uint8)

        img = Image.fromarray(out_arr, 'L')
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img.save(filepath, format="PNG")

    @staticmethod
    def save_depth_npy(filepath: str, depth_data: np.ndarray, width: int, height: int) -> None:
        depth_map = np.array(depth_data, dtype=np.float32).reshape((height, width))
        depth_map = np.flipud(depth_map)
        np.save(filepath, depth_map)