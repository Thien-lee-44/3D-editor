from typing import List, Dict, Any

class YOLOWriter:
    """
    Serializes object data into standard YOLO dataset formats.
    Dynamically supports both Object Detection (Center Bounding Box) 
    and Instance Segmentation (Polygon Contours).
    Enforces strict domain normalization [0.0, 1.0] and spatial validation.
    """

    @staticmethod
    def export(filepath: str, objects: List[Dict[str, Any]], img_w: int, img_h: int, is_segmentation: bool = False) -> None:
        """
        Exports a YOLO annotation file.
        
        :param filepath: Destination file path (.txt).
        :param objects: List of detected objects containing 'class_id', 'bbox_xyxy', and optionally 'segmentation'.
        :param img_w: Source image width for normalization.
        :param img_h: Source image height for normalization.
        :param is_segmentation: Toggles export mode between Segmentation (Polygons) and Detection (BBoxes).
        """
        lines = []
        
        for obj in objects:
            class_id = int(obj.get("class_id", 0))
            
            if is_segmentation:
                # ---------------------------------------------------------
                # MODE: YOLO INSTANCE SEGMENTATION (Polygons)
                # Format: <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
                # ---------------------------------------------------------
                polygons = obj.get("segmentation", [])
                
                if polygons and len(polygons[0]) >= 6:
                    poly = polygons[0]
                    normalized_poly = []
                    for i in range(0, len(poly), 2):
                        nx = max(0.0, min(1.0, poly[i] / img_w))
                        ny = max(0.0, min(1.0, poly[i+1] / img_h))
                        normalized_poly.extend([nx, ny])
                    
                    poly_str = " ".join([f"{v:.6f}" for v in normalized_poly])
                    lines.append(f"{class_id} {poly_str}")
                    
                else:
                    # Fallback: Extrapolate a 4-point polygon from the bounding box
                    bbox = obj.get("bbox_xyxy")
                    if bbox and len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        nx1 = max(0.0, min(1.0, x1 / img_w))
                        ny1 = max(0.0, min(1.0, y1 / img_h))
                        nx2 = max(0.0, min(1.0, x2 / img_w))
                        ny2 = max(0.0, min(1.0, y2 / img_h))
                        
                        if abs(nx2 - nx1) > 1e-4 and abs(ny2 - ny1) > 1e-4:
                            rect = [nx1, ny1, nx2, ny1, nx2, ny2, nx1, ny2]
                            poly_str = " ".join([f"{v:.6f}" for v in rect])
                            lines.append(f"{class_id} {poly_str}")
            else:
                # ---------------------------------------------------------
                # MODE: YOLO OBJECT DETECTION (Bounding Boxes)
                # Format: <class_id> <x_center> <y_center> <width> <height>
                # ---------------------------------------------------------
                bbox = obj.get("bbox_xyxy")
                if not bbox or len(bbox) != 4:
                    continue
                    
                xmin, ymin, xmax, ymax = bbox
                
                abs_w = max(0.0, xmax - xmin)
                abs_h = max(0.0, ymax - ymin)
                abs_cx = xmin + (abs_w / 2.0)
                abs_cy = ymin + (abs_h / 2.0)

                norm_cx = max(0.0, min(1.0, abs_cx / img_w))
                norm_cy = max(0.0, min(1.0, abs_cy / img_h))
                norm_w = max(0.0, min(1.0, abs_w / img_w))
                norm_h = max(0.0, min(1.0, abs_h / img_h))

                # Discard ghost bounding boxes to prevent division-by-zero crashes during AI training
                if norm_w <= 0.0001 or norm_h <= 0.0001:
                    continue

                lines.append(f"{class_id} {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}")

        # Always write the file, even if empty. 
        # YOLO requires empty annotation files for background/negative sample training.
        with open(filepath, 'w', encoding='utf-8') as f:
            if lines:
                f.write("\n".join(lines) + "\n")
            else:
                f.write("")