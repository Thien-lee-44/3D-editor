import numpy as np
from PIL import Image

class ImageWriter:
    """
    Handles the serialization of raw OpenGL pixel buffers to standard image formats.
    Automatically rectifies OpenGL's bottom-left origin to the standard top-left origin.
    """

    @staticmethod
    def save_rgb(filepath: str, pixel_data: bytes, width: int, height: int) -> None:
        """Saves a standard 3-channel RGB image (High Quality JPEG to save disk space)."""
        arr = np.frombuffer(pixel_data, dtype=np.uint8).reshape((height, width, 3))
        img = Image.fromarray(arr, 'RGB')
        
        # OpenGL reads from Bottom-Left. Images require Top-Left origin.
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        
        # Use 100% quality to minimize compression artifacts for AI training
        img.save(filepath, format="JPEG", quality=100)

    @staticmethod
    def save_mask(filepath: str, pixel_data: bytes, width: int, height: int) -> None:
        """
        Saves a 3-channel Instance/Semantic Mask image.
        MUST be a lossless PNG to strictly preserve the exact RGB Semantic ID values.
        """
        arr = np.frombuffer(pixel_data, dtype=np.uint8).reshape((height, width, 3))
        img = Image.fromarray(arr, 'RGB')
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img.save(filepath, format="PNG")

    @staticmethod
    def save_depth(filepath: str, depth_data: np.ndarray, width: int, height: int) -> None:
        """
        Saves a 1-channel Depth Map. 
        Normalizes the non-linear OpenGL depth buffer for visual perception.
        """
        # 1. Ignore background (1.0 in OpenGL depth means infinitely far away)
        valid_mask = depth_data < 1.0
        
        if np.any(valid_mask):
            d_min = depth_data[valid_mask].min()
            d_max = depth_data[valid_mask].max()
            
            if d_max > d_min:
                # Normalize the valid depths to span the full 0.0 to 1.0 range
                normalized = np.zeros_like(depth_data)
                normalized[valid_mask] = (depth_data[valid_mask] - d_min) / (d_max - d_min)
                
                # Invert depth so closer objects appear brighter (Standard CV practice)
                normalized[valid_mask] = 1.0 - normalized[valid_mask]
                
                out_arr = (normalized * 255).astype(np.uint8)
            else:
                out_arr = np.zeros_like(depth_data, dtype=np.uint8)
        else:
            out_arr = np.zeros_like(depth_data, dtype=np.uint8)

        img = Image.fromarray(out_arr, 'L')
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img.save(filepath, format="PNG")