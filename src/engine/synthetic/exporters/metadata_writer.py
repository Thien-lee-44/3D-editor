import csv
import json
from pathlib import Path
from typing import Any, Dict, List


class MetadataWriter:
    """
    Writes structured scene metadata for each generated frame.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._frames: List[Dict[str, Any]] = []
        self._object_rows: List[Dict[str, Any]] = []

    def add_frame(self, frame_record: Dict[str, Any]) -> None:
        self._frames.append(frame_record)

        frame_index = int(frame_record.get("frame_index", -1))
        files = frame_record.get("files", {})
        camera = frame_record.get("camera", {})

        for obj in frame_record.get("objects", []):
            bbox = obj.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])
            self._object_rows.append(
                {
                    "frame_index": frame_index,
                    "image": files.get("image", ""),
                    "label": files.get("label", ""),
                    "track_id": int(obj.get("track_id", -1)),
                    "class_id": int(obj.get("class_id", 0)),
                    "class_name": obj.get("class_name", "Unknown"),
                    "entity_name": obj.get("entity_name", "unknown"),
                    "bbox_xmin": float(bbox[0]),
                    "bbox_ymin": float(bbox[1]),
                    "bbox_xmax": float(bbox[2]),
                    "bbox_ymax": float(bbox[3]),
                    "visible_pixels": int(obj.get("visible_pixels", 0)),
                    "visibility_ratio": float(obj.get("visibility_ratio", 0.0)),
                    "occlusion_ratio": float(obj.get("occlusion_ratio", 0.0)),
                    "object_world_x": float(obj.get("world_position", [0.0, 0.0, 0.0])[0]),
                    "object_world_y": float(obj.get("world_position", [0.0, 0.0, 0.0])[1]),
                    "object_world_z": float(obj.get("world_position", [0.0, 0.0, 0.0])[2]),
                    "camera_world_x": float(camera.get("position", [0.0, 0.0, 0.0])[0]),
                    "camera_world_y": float(camera.get("position", [0.0, 0.0, 0.0])[1]),
                    "camera_world_z": float(camera.get("position", [0.0, 0.0, 0.0])[2]),
                }
            )

    def flush(self) -> None:
        json_path = self.output_dir / "frames.json"
        ndjson_path = self.output_dir / "frames.ndjson"
        csv_path = self.output_dir / "objects.csv"

        payload = {"frames": self._frames}
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        with open(ndjson_path, "w", encoding="utf-8") as f:
            for frame in self._frames:
                f.write(json.dumps(frame, ensure_ascii=False) + "\n")

        csv_columns = [
            "frame_index",
            "image",
            "label",
            "track_id",
            "class_id",
            "class_name",
            "entity_name",
            "bbox_xmin",
            "bbox_ymin",
            "bbox_xmax",
            "bbox_ymax",
            "visible_pixels",
            "visibility_ratio",
            "occlusion_ratio",
            "object_world_x",
            "object_world_y",
            "object_world_z",
            "camera_world_x",
            "camera_world_y",
            "camera_world_z",
        ]

        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writeheader()
            if self._object_rows:
                writer.writerows(self._object_rows)
