"""
COCO Dataset Format Exporter.

Aggregates frame annotations and exports a single COCO JSON file.
Complies with the standard Common Objects in Context (COCO) specifications.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List


class COCOWriter:
    """Maintains COCO state across multiple frames and flushes to a single JSON payload."""

    def __init__(self, output_path: Path, categories: Dict[int, Any]) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.images: List[Dict[str, Any]] = []
        self.annotations: List[Dict[str, Any]] = []
        self.categories: List[Dict[str, Any]] = self._normalize_categories(categories)
        self._next_annotation_id: int = 1

    @staticmethod
    def _normalize_categories(categories: Dict[int, Any]) -> List[Dict[str, Any]]:
        """Formats semantic classes into the COCO category schema."""
        normalized: List[Dict[str, Any]] = []

        for raw_id, raw_info in sorted(categories.items(), key=lambda item: int(item[0])):
            class_id = int(raw_id)

            if isinstance(raw_info, dict):
                name = str(raw_info.get("name", f"class_{class_id}"))
            else:
                name = str(raw_info)

            normalized.append(
                {
                    "id": class_id,
                    "name": name,
                    "supercategory": "object",
                }
            )

        return normalized

    def add_frame(
        self,
        frame_index: int,
        image_file: str,
        width: int,
        height: int,
        objects: Iterable[Dict[str, Any]],
    ) -> None:
        """Registers a single image frame and all of its associated object bounds."""
        image_id = frame_index + 1

        self.images.append(
            {
                "id": image_id,
                "file_name": image_file,
                "width": int(width),
                "height": int(height),
            }
        )

        for obj in objects:
            bbox_xyxy = obj.get("bbox_xyxy", None)
            if not bbox_xyxy or len(bbox_xyxy) != 4:
                continue

            xmin, ymin, xmax, ymax = [float(v) for v in bbox_xyxy]
            box_w = max(0.0, xmax - xmin)
            box_h = max(0.0, ymax - ymin)
            
            if box_w <= 0.0 or box_h <= 0.0:
                continue

            visible_pixels = int(obj.get("visible_pixels", int(box_w * box_h)))

            raw_segmentation = obj.get("segmentation", None)
            if isinstance(raw_segmentation, list) and raw_segmentation:
                segmentation = []
                for polygon in raw_segmentation:
                    if not isinstance(polygon, list) or len(polygon) < 6:
                        continue
                    segmentation.append([float(v) for v in polygon])
                if not segmentation:
                    segmentation = [[xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]]
            else:
                # Rectangular polygon fallback keeps COCO segmentation field valid.
                segmentation = [[xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]]

            self.annotations.append(
                {
                    "id": self._next_annotation_id,
                    "image_id": image_id,
                    "category_id": int(obj.get("class_id", 0)),
                    "bbox": [xmin, ymin, box_w, box_h],
                    "area": float(max(visible_pixels, box_w * box_h)),
                    "iscrowd": 0,
                    "segmentation": segmentation,
                    "track_id": int(obj.get("track_id", -1)),
                }
            )
            self._next_annotation_id += 1

    def flush(self) -> None:
        """Writes the accumulated COCO dataset payload to disk."""
        payload = {
            "info": {
                "description": "Synthetic dataset exported from generator",
                "version": "1.0",
                "year": datetime.now().year,
                "date_created": datetime.now().isoformat(timespec="seconds"),
            },
            "licenses": [],
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories,
        }

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)