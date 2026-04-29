"""
Label Extraction Utilities.

Provides high-performance NumPy algorithms to process raw rendered masks
and extract bounding boxes or metadata for AI datasets.
"""

import numpy as np
from typing import Any, Dict


class LabelUtils:
    """Utility class for parsing Ground Truth pixel data."""

    @staticmethod
    def extract_bboxes_from_mask(mask_pixels: bytes, width: int, height: int) -> Dict[int, Dict[str, Any]]:
        """
        Decodes an RGB instance mask into discrete bounding box coordinates.
        Reconstructs the 24-bit integer Track ID encoded across the R, G, B channels.

        Args:
            mask_pixels: Raw byte array of the rendered RGB mask.
            width: Resolution width.
            height: Resolution height.

        Returns:
            A dictionary mapping Track IDs to their bounding box data and visible pixel count.
        """
        if not mask_pixels:
            return {}
            
        arr = np.frombuffer(mask_pixels, dtype=np.uint8).reshape((height, width, 3))
        
        # Reconstruct the original 24-bit integer ID using Bitwise Shift
        id_map = arr[:, :, 0].astype(np.uint32) | \
                 (arr[:, :, 1].astype(np.uint32) << 8) | \
                 (arr[:, :, 2].astype(np.uint32) << 16)

        unique_ids = np.unique(id_map)
        bboxes = {}

        for uid in unique_ids:
            # ID 0 is strictly reserved for the unannotated background (Black pixels)
            if uid == 0: 
                continue 
            
            mask = (id_map == uid)
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            
            visible_pixels = int(np.sum(mask))

            bboxes[int(uid)] = {
                "bbox": (float(xmin), float(ymin), float(xmax), float(ymax)),
                "visible_pixels": visible_pixels
            }

        return bboxes