"""
Pascal VOC Format Exporter.

Exports object detection annotations in the standard XML format used by Pascal VOC.
"""

from pathlib import Path
from typing import Any, Dict, Iterable
import xml.etree.ElementTree as ET


class VOCWriter:
    """Generates Pascal VOC XML annotations."""

    @staticmethod
    def export(
        filepath: str,
        image_file: str,
        width: int,
        height: int,
        objects: Iterable[Dict[str, Any]],
        depth: int = 3,
    ) -> None:
        """
        Serializes detected object bounding boxes into a formatted XML file.
        
        Args:
            filepath: Destination file path (.xml).
            image_file: The relative or absolute path to the corresponding image file.
            width: Width of the associated image.
            height: Height of the associated image.
            objects: Iterable of objects containing 'bbox_xyxy' and 'class_name'/'class_id'.
            depth: Number of color channels in the image (default is 3 for RGB).
        """
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        annotation = ET.Element("annotation")
        ET.SubElement(annotation, "folder").text = str(Path(image_file).parent.as_posix())
        ET.SubElement(annotation, "filename").text = str(Path(image_file).name)
        ET.SubElement(annotation, "path").text = str(image_file)

        source = ET.SubElement(annotation, "source")
        ET.SubElement(source, "database").text = "Synthetic BTL2"

        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(int(width))
        ET.SubElement(size, "height").text = str(int(height))
        ET.SubElement(size, "depth").text = str(int(depth))
        ET.SubElement(annotation, "segmented").text = "0"

        for obj in objects:
            bbox = obj.get("bbox_xyxy")
            if not bbox or len(bbox) != 4:
                continue

            xmin, ymin, xmax, ymax = [float(v) for v in bbox]
            if xmax <= xmin or ymax <= ymin:
                continue

            class_name = str(obj.get("class_name", f"class_{int(obj.get('class_id', 0))}"))

            item = ET.SubElement(annotation, "object")
            ET.SubElement(item, "name").text = class_name
            ET.SubElement(item, "pose").text = "Unspecified"
            ET.SubElement(item, "truncated").text = "0"
            ET.SubElement(item, "difficult").text = "0"

            bndbox = ET.SubElement(item, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(int(round(max(0.0, xmin))))
            ET.SubElement(bndbox, "ymin").text = str(int(round(max(0.0, ymin))))
            ET.SubElement(bndbox, "xmax").text = str(int(round(min(float(width), xmax))))
            ET.SubElement(bndbox, "ymax").text = str(int(round(min(float(height), ymax))))

        tree = ET.ElementTree(annotation)
        tree.write(str(output_path), encoding="utf-8", xml_declaration=True)