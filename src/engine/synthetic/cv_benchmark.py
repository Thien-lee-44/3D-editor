"""
Computer Vision Benchmark Utility.

Automates the training, validation, and benchmarking of YOLO models on synthetic datasets.
Dynamically handles Train/Val/Test splits via label lists, preventing data duplication.
Generates performance metrics and visual comparison artifacts.
"""

import csv
import json
import os
import shutil
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageDraw

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class CVBenchmarkConfig:
    model_type: Optional[str] = None
    task: str = "auto"
    epochs: int = 3
    batch_size: int = 8
    imgsz: int = 640
    confidence_threshold: float = 0.25
    run_training: bool = True
    iou_match_threshold: float = 0.5
    max_visualization_frames: int = 0
    split_ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1)

class CVBenchmarkRunner:
    """
    Executes reproducible CV benchmarks across dataset variants.
    Aggregates performance metrics and generates comparative visual artifacts.
    """

    def __init__(self, output_dir: Path, config: Optional[CVBenchmarkConfig] = None) -> None:
        self.output_dir = Path(output_dir)
        self.config = config or CVBenchmarkConfig()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "yolo_runs").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "benchmark_visuals").mkdir(parents=True, exist_ok=True)

    def _create_benchmark_splits(self, dataset_dir: Path, original_yaml: Path, safe_name: str) -> Path:
        """
        Dynamically partitions the flat dataset into train, val, and test splits 
        using text files containing absolute paths. This satisfies YOLO's requirements 
        without duplicating images on disk.
        """
        split_dir = self.output_dir / "splits" / safe_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        images = self._iter_images(dataset_dir / "images")
        
        # Use a fixed seed to ensure reproducibility of the benchmark on the same data
        random.seed(42)
        random.shuffle(images)
        
        total = len(images)
        ratios = self.config.split_ratios
        train_end = int(total * ratios[0])
        val_end = train_end + int(total * ratios[1])
        
        train_imgs = images[:train_end]
        val_imgs = images[train_end:val_end]
        test_imgs = images[val_end:]
        
        # Ensure fallback if requested split is empty
        if not train_imgs and images: train_imgs = images
        if not val_imgs and images: val_imgs = train_imgs
        
        train_txt = split_dir / "train.txt"
        val_txt = split_dir / "val.txt"
        test_txt = split_dir / "test.txt"
        
        train_txt.write_text("\n".join(str(p.absolute()) for p in train_imgs))
        val_txt.write_text("\n".join(str(p.absolute()) for p in val_imgs))
        if test_imgs:
            test_txt.write_text("\n".join(str(p.absolute()) for p in test_imgs))
            
        # Parse original YAML for class names
        class_names = {}
        try:
            with open(original_yaml, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                class_names = data.get("names", {})
        except Exception:
            pass

        # Generate a temporary benchmark YAML
        new_yaml = split_dir / "dataset_benchmark.yaml"
        yaml_content = [
            f"path: {dataset_dir.absolute().as_posix()}",
            f"train: {train_txt.absolute().as_posix()}",
            f"val: {val_txt.absolute().as_posix()}",
        ]
        
        if test_imgs:
            yaml_content.append(f"test: {test_txt.absolute().as_posix()}")
            
        yaml_content.append(f"nc: {len(class_names)}")
        yaml_content.append("names:")
        
        if isinstance(class_names, dict):
            for k, v in class_names.items():
                yaml_content.append(f"  {k}: \"{v}\"")
        elif isinstance(class_names, list):
            for i, v in enumerate(class_names):
                yaml_content.append(f"  {i}: \"{v}\"")

        new_yaml.write_text("\n".join(yaml_content) + "\n", encoding="utf-8")
        return new_yaml

    def run(self, variant_dirs: Dict[str, Path], progress_cb: Optional[Callable[[str, int, int], None]] = None) -> Dict[str, Any]:
        if YOLO is None:
            raise RuntimeError("Ultralytics YOLO is not installed.")

        records: List[Dict[str, Any]] = []
        items = list(variant_dirs.items())
        total = len(items)

        for idx, (variant_name, variant_dir_raw) in enumerate(items, start=1):
            if progress_cb:
                progress_cb(variant_name, idx, total)

            variant_dir = Path(variant_dir_raw)
            dataset_yaml = variant_dir / "dataset.yaml"
            safe_name = self._safe_name(variant_name)

            if not dataset_yaml.exists():
                records.append({"variant": variant_name, "status": "missing_dataset_yaml"})
                continue

            try:
                task = self._resolve_task(variant_dir)
                model_type = self._resolve_model_type(task)
            except Exception as exc:
                records.append({"variant": variant_name, "status": "error", "error": str(exc)})
                continue

            effective_variant_dir = variant_dir
            effective_dataset_yaml = dataset_yaml

            try:
                if task == "segment":
                    effective_variant_dir, effective_dataset_yaml = self._prepare_segment_variant(variant_dir, safe_name)

                # =============================================================
                # DYNAMIC TRAIN/VAL/TEST SPLITTING
                # =============================================================
                benchmark_yaml = self._create_benchmark_splits(effective_variant_dir, effective_dataset_yaml, safe_name)

                project_dir = self.output_dir / "yolo_runs" / safe_name
                model = YOLO(model_type)

                if self.config.run_training:
                    device = "0" if torch.cuda.is_available() else "cpu"
                    workers = 8 if torch.cuda.is_available() else 2
                    if task == "segment": workers = 0

                    train_kwargs: Dict[str, Any] = {
                        "data": str(benchmark_yaml), # Use dynamically generated split config
                        "epochs": int(self.config.epochs),
                        "batch": int(self.config.batch_size),
                        "imgsz": int(self.config.imgsz),
                        "project": str(project_dir),
                        "name": "train",
                        "exist_ok": True,
                        "device": device,
                        "workers": workers,
                        "patience": 5,
                        "verbose": False,
                        "plots": False,
                    }
                    if task == "segment": train_kwargs["copy_paste"] = 0.0

                    model.train(**train_kwargs)

                    best_weights = project_dir / "train" / "weights" / "best.pt"
                    if best_weights.exists():
                        model = YOLO(str(best_weights))

                # Evaluate on Test split if it exists, otherwise Val
                has_test_split = False
                with open(benchmark_yaml, 'r', encoding='utf-8') as f:
                    has_test_split = "test:" in f.read()
                
                eval_split = "test" if has_test_split else "val"

                metrics_out = model.val(
                    data=str(benchmark_yaml),
                    project=str(project_dir),
                    name=f"eval_{eval_split}",
                    exist_ok=True,
                    verbose=False,
                    split=eval_split,
                )

                metrics = {
                    "box_map50": float(metrics_out.box.map50) if hasattr(metrics_out, "box") else 0.0,
                    "box_map50_95": float(metrics_out.box.map) if hasattr(metrics_out, "box") else 0.0,
                    "box_precision": float(metrics_out.box.mp) if hasattr(metrics_out, "box") else 0.0,
                    "box_recall": float(metrics_out.box.mr) if hasattr(metrics_out, "box") else 0.0,
                    "seg_map50": float(metrics_out.seg.map50) if hasattr(metrics_out, "seg") else 0.0,
                    "seg_map50_95": float(metrics_out.seg.map) if hasattr(metrics_out, "seg") else 0.0,
                    "seg_precision": float(metrics_out.seg.mp) if hasattr(metrics_out, "seg") else 0.0,
                    "seg_recall": float(metrics_out.seg.mr) if hasattr(metrics_out, "seg") else 0.0,
                }
                primary_metric_name = "seg_map50" if task == "segment" else "box_map50"

                pred_dir = str(project_dir / "predict_all")
                model.predict(
                    source=str(effective_variant_dir / "images"),
                    conf=float(self.config.confidence_threshold),
                    project=str(project_dir),
                    name="predict_all",
                    save=True,
                    exist_ok=True,
                    verbose=False,
                )

                class_lookup = self._load_class_names(benchmark_yaml)
                comparison_artifacts = self._export_comparison_artifacts(
                    model=model,
                    variant_name=safe_name,
                    variant_dir=effective_variant_dir,
                    class_lookup=class_lookup,
                    task=task,
                )

                img_count, lbl_count = self._dataset_counts(effective_variant_dir)

                records.append({
                    "variant": variant_name,
                    "status": "ok",
                    "dataset_dir": str(variant_dir),
                    "dataset_yaml": str(benchmark_yaml),
                    "task": task,
                    "model_type": model_type,
                    "eval_split": eval_split,
                    "num_images": img_count,
                    "num_labels": lbl_count,
                    "primary_metric_name": primary_metric_name,
                    "primary_metric": float(metrics.get(primary_metric_name, 0.0)),
                    "metrics": metrics,
                    "prediction_dir": pred_dir,
                    "comparison_dir": comparison_artifacts.get("comparison_dir", ""),
                    "comparison_csv": comparison_artifacts.get("comparison_csv", ""),
                    "comparison_json": comparison_artifacts.get("comparison_json", ""),
                    "visualized_frames": int(comparison_artifacts.get("visualized_frames", 0)),
                    "dataset_total_frames": int(comparison_artifacts.get("dataset_total_frames", 0)),
                    "match_precision": float(comparison_artifacts.get("match_precision", 0.0)),
                    "match_recall": float(comparison_artifacts.get("match_recall", 0.0)),
                })
                
            except Exception as exc:
                records.append({"variant": variant_name, "status": "error", "error": str(exc)})

        artifacts = self._write_artifacts(records)
        return {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "records": records,
            "artifacts": artifacts,
        }

    def _write_artifacts(self, records: List[Dict[str, Any]]) -> Dict[str, str]:
        """Dumps aggregated performance results into JSON, CSV, and Markdown summaries."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_json = self.output_dir / f"cv_benchmark_metrics_{timestamp}.json"
        metrics_csv = self.output_dir / f"cv_benchmark_metrics_{timestamp}.csv"
        summary_md = self.output_dir / f"cv_benchmark_summary_{timestamp}.md"

        # Export JSON
        with open(metrics_json, "w", encoding="utf-8") as f:
            json.dump({"records": records}, f, ensure_ascii=False, indent=2)

        # Export CSV
        csv_columns = [
            "variant", "status", "dataset_dir", "dataset_yaml", "effective_dataset_yaml",
            "task", "model_type", "eval_split", "num_images", "num_labels", "primary_metric_name", "primary_metric",
            "box_map50", "box_map50_95", "box_precision", "box_recall",
            "seg_map50", "seg_map50_95", "seg_precision", "seg_recall",
            "match_precision", "match_recall", "dataset_total_frames", "visualized_frames",
            "prediction_dir", "comparison_dir", "comparison_csv", "comparison_json", "error"
        ]
        
        with open(metrics_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writeheader()
            for rec in records:
                m = rec.get("metrics", {})
                row_data = {k: rec.get(k, "") for k in csv_columns if k not in ["box_map50", "box_map50_95", "box_precision", "box_recall", "seg_map50", "seg_map50_95", "seg_precision", "seg_recall"]}
                
                # Merge metric sub-dictionary safely
                row_data.update({
                    "primary_metric": float(rec.get("primary_metric", 0.0)),
                    "num_images": rec.get("num_images", 0),
                    "num_labels": rec.get("num_labels", 0),
                    "dataset_total_frames": int(rec.get("dataset_total_frames", 0)),
                    "visualized_frames": int(rec.get("visualized_frames", 0)),
                    "match_precision": float(rec.get("match_precision", 0.0)),
                    "match_recall": float(rec.get("match_recall", 0.0)),
                    "box_map50": float(m.get("box_map50", 0.0)),
                    "box_map50_95": float(m.get("box_map50_95", 0.0)),
                    "box_precision": float(m.get("box_precision", 0.0)),
                    "box_recall": float(m.get("box_recall", 0.0)),
                    "seg_map50": float(m.get("seg_map50", 0.0)),
                    "seg_map50_95": float(m.get("seg_map50_95", 0.0)),
                    "seg_precision": float(m.get("seg_precision", 0.0)),
                    "seg_recall": float(m.get("seg_recall", 0.0)),
                })
                writer.writerow(row_data)

        # Export Markdown Summary
        with open(summary_md, "w", encoding="utf-8") as f:
            f.write("# CV Benchmark Summary\n\n")
            f.write(f"Generated at: {datetime.now().isoformat(timespec='seconds')}\n\n")

            ok_records = [r for r in records if r.get("status") == "ok"]
            ranked = sorted(ok_records, key=lambda r: float(r.get("primary_metric", 0.0)), reverse=True)

            if not ranked:
                f.write("No successful benchmark run was produced.\n")
            else:
                metric_name = str(ranked[0].get("primary_metric_name", "box_map50"))
                f.write(f"## Ranking by {metric_name}\n\n")
                for i, rec in enumerate(ranked, start=1):
                    m = rec.get("metrics", {})
                    task = rec.get("task", "detect")
                    f.write(
                        f"{i}. **{rec.get('variant', 'Unknown')}** - "
                        f"Task: {task} (eval: {rec.get('eval_split', 'val')}) | "
                        f"{metric_name}: **{float(rec.get('primary_metric', 0.0)):.4f}** | "
                        f"Box mAP@50: {float(m.get('box_map50', 0.0)):.4f} | "
                        f"Seg mAP@50: {float(m.get('seg_map50', 0.0)):.4f}\n"
                    )

                f.write("\n## Detail Performance Table\n\n")
                f.write("| Variant | Task | Imgs | Lbls | Eval | Primary | Box mAP@50 | Seg mAP@50 | Match P | Match R | Visualized/Total |\n")
                f.write("|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|\n")
                for rec in ranked:
                    m = rec.get("metrics", {})
                    f.write(
                        f"| {rec.get('variant', 'Unknown')} "
                        f"| {rec.get('task', '')} "
                        f"| {int(rec.get('num_images', 0))} "
                        f"| {int(rec.get('num_labels', 0))} "
                        f"| {rec.get('eval_split', '')} "
                        f"| **{float(rec.get('primary_metric', 0.0)):.4f}** "
                        f"| {float(m.get('box_map50', 0.0)):.4f} "
                        f"| {float(m.get('seg_map50', 0.0)):.4f} "
                        f"| {float(rec.get('match_precision', 0.0)):.4f} "
                        f"| {float(rec.get('match_recall', 0.0)):.4f} "
                        f"| {int(rec.get('visualized_frames', 0))}/{int(rec.get('dataset_total_frames', 0))} |\n"
                    )

            failed = [r for r in records if r.get("status") != "ok"]
            if failed:
                f.write("\n## Failed Variants\n\n")
                for rec in failed:
                    f.write(f"- {rec.get('variant', 'Unknown')}: {rec.get('status', 'unknown')}\n")
                    if rec.get("error"):
                        f.write(f"  - Error: {rec['error']}\n")

        return {
            "json": str(metrics_json),
            "csv": str(metrics_csv),
            "summary_md": str(summary_md),
        }

    def _export_comparison_artifacts(self, model: Any, variant_name: str, variant_dir: Path, class_lookup: Dict[int, str], task: str) -> Dict[str, Any]:
        """Generates side-by-side Ground Truth vs. Prediction image artifacts for qualitative analysis."""
        image_dir = variant_dir / "images"
        label_dir = variant_dir / "labels"
        image_paths = self._iter_images(image_dir)
        total_frames = len(image_paths)
        
        if total_frames == 0:
            return {
                "comparison_dir": "", "comparison_csv": "", "comparison_json": "",
                "visualized_frames": 0, "dataset_total_frames": 0,
                "match_precision": 0.0, "match_recall": 0.0,
            }

        max_frames = int(self.config.max_visualization_frames)
        selected_paths = image_paths if max_frames <= 0 else image_paths[:max_frames]

        comparison_dir = self.output_dir / "benchmark_visuals" / variant_name
        if comparison_dir.exists():
            shutil.rmtree(comparison_dir, ignore_errors=True)
        comparison_dir.mkdir(parents=True, exist_ok=True)

        frame_rows: List[Dict[str, Any]] = []
        sum_gt = sum_pred = sum_matched = 0

        for frame_idx, image_path in enumerate(selected_paths):
            with Image.open(image_path) as pil_img:
                img = pil_img.convert("RGB")
                
            width, height = img.size
            
            # Recursively resolve label path relative to image path
            try:
                rel_path = image_path.relative_to(image_dir)
                label_path = label_dir / rel_path.parent / f"{image_path.stem}.txt"
            except ValueError:
                label_path = label_dir / f"{image_path.stem}.txt"
            
            gt_boxes = self._read_gt_boxes(label_path, width, height, class_lookup)
            pred_boxes = self._predict_boxes(model, image_path, class_lookup)
            matched = self._count_matches(gt_boxes, pred_boxes, float(self.config.iou_match_threshold))
            
            gt_count, pred_count = len(gt_boxes), len(pred_boxes)
            sum_gt += gt_count
            sum_pred += pred_count
            sum_matched += matched

            frame_precision = (matched / pred_count) if pred_count > 0 else (1.0 if gt_count == 0 else 0.0)
            frame_recall = (matched / gt_count) if gt_count > 0 else 1.0

            out_img = comparison_dir / f"{image_path.stem}_compare.jpg"
            self._render_comparison_image(
                image=img, gt_boxes=gt_boxes, pred_boxes=pred_boxes,
                output_path=out_img, task=task,
            )

            frame_rows.append({
                "frame_index": frame_idx, "image": str(image_path.name),
                "gt_count": gt_count, "pred_count": pred_count,
                "matched_count": matched, "precision_like": frame_precision,
                "recall_like": frame_recall, "comparison_image": str(out_img.name),
            })

        overall_precision = (sum_matched / sum_pred) if sum_pred > 0 else (1.0 if sum_gt == 0 else 0.0)
        overall_recall = (sum_matched / sum_gt) if sum_gt > 0 else 1.0

        comparison_csv = comparison_dir / "benchmark_frame_comparison.csv"
        with open(comparison_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "frame_index", "image", "gt_count", "pred_count", "matched_count",
                "precision_like", "recall_like", "comparison_image"
            ])
            writer.writeheader()
            writer.writerows(frame_rows)

        comparison_json = comparison_dir / "benchmark_comparison_summary.json"
        with open(comparison_json, "w", encoding="utf-8") as f:
            json.dump({
                "variant": variant_name, "task": task, "dataset_total_frames": total_frames,
                "visualized_frames": len(selected_paths), "sum_gt": sum_gt, "sum_pred": sum_pred,
                "sum_matched": sum_matched, "match_precision": overall_precision, "match_recall": overall_recall,
                "frames": frame_rows,
            }, f, ensure_ascii=False, indent=2)

        return {
            "comparison_dir": str(comparison_dir), "comparison_csv": str(comparison_csv),
            "comparison_json": str(comparison_json), "visualized_frames": len(selected_paths),
            "dataset_total_frames": total_frames, "match_precision": overall_precision,
            "match_recall": overall_recall,
        }

    def _render_comparison_image(self, image: Image.Image, gt_boxes: List[Dict[str, Any]], pred_boxes: List[Dict[str, Any]], output_path: Path, task: str) -> None:
        """Draws bounding boxes and labels onto the side-by-side artifact image."""
        left, right = image.copy(), image.copy()
        draw_left, draw_right = ImageDraw.Draw(left), ImageDraw.Draw(right)

        for item in gt_boxes:
            x1, y1, x2, y2 = item["bbox_xyxy"]
            name = item.get("class_name", "GT")
            draw_left.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
            draw_left.text((x1 + 3, max(0, y1 - 14)), f"GT:{name}", fill=(0, 255, 0))

        for item in pred_boxes:
            x1, y1, x2, y2 = item["bbox_xyxy"]
            name = item.get("class_name", "Pred")
            conf = float(item.get("confidence", 0.0))
            draw_right.rectangle([x1, y1, x2, y2], outline=(255, 80, 80), width=2)
            draw_right.text((x1 + 3, max(0, y1 - 14)), f"PD:{name} {conf:.2f}", fill=(255, 80, 80))

        title_h = 24
        pad = 4
        canvas = Image.new("RGB", (left.width * 2 + pad * 3, left.height + title_h + pad * 2), color=(22, 22, 22))
        canvas.paste(left, (pad, pad + title_h))
        canvas.paste(right, (left.width + pad * 2, pad + title_h))

        draw = ImageDraw.Draw(canvas)
        draw.text((pad + 6, 6), f"Ground Truth ({task})", fill=(0, 255, 0))
        draw.text((left.width + pad * 2 + 6, 6), "Prediction", fill=(255, 80, 80))

        canvas.save(output_path, format="JPEG", quality=95)

    def _predict_boxes(self, model: Any, image_path: Path, class_lookup: Dict[int, str]) -> List[Dict[str, Any]]:
        """Invokes the YOLO model to detect bounding boxes on a target image."""
        results = model.predict(
            source=str(image_path), conf=float(self.config.confidence_threshold),
            imgsz=int(self.config.imgsz), save=False, verbose=False,
        )
        if not results:
            return []

        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        out: List[Dict[str, Any]] = []
        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            cls_id = int(float(box.cls[0]))
            conf = float(box.conf[0]) if getattr(box, "conf", None) is not None else 0.0
            out.append({
                "class_id": cls_id,
                "class_name": class_lookup.get(cls_id, f"class_{cls_id}"),
                "confidence": conf,
                "bbox_xyxy": [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])],
            })
        return out

    def _read_gt_boxes(self, label_file: Path, width: int, height: int, class_lookup: Dict[int, str]) -> List[Dict[str, Any]]:
        """Parses Ground Truth YOLO labels (.txt) and converts normalized coordinates back to absolute pixels."""
        if not label_file.exists():
            return []

        boxes: List[Dict[str, Any]] = []
        with open(label_file, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                    
                parts = line.split()
                if len(parts) < 5:
                    continue

                try:
                    cls_id = int(float(parts[0]))
                except ValueError:
                    continue

                if len(parts) == 5:
                    try:
                        cx, cy = float(parts[1]) * width, float(parts[2]) * height
                        bw, bh = float(parts[3]) * width, float(parts[4]) * height
                    except ValueError:
                        continue
                        
                    x1 = max(0.0, cx - bw / 2.0)
                    y1 = max(0.0, cy - bh / 2.0)
                    x2 = min(float(width), cx + bw / 2.0)
                    y2 = min(float(height), cy + bh / 2.0)
                else:
                    coords: List[Tuple[float, float]] = []
                    try:
                        for i in range(1, len(parts), 2):
                            coords.append((float(parts[i]) * width, float(parts[i + 1]) * height))
                    except (ValueError, IndexError):
                        continue

                    if not coords:
                        continue
                        
                    xs, ys = [p[0] for p in coords], [p[1] for p in coords]
                    x1 = max(0.0, min(xs))
                    y1 = max(0.0, min(ys))
                    x2 = min(float(width), max(xs))
                    y2 = min(float(height), max(ys))

                if x2 <= x1 or y2 <= y1:
                    continue

                boxes.append({
                    "class_id": cls_id,
                    "class_name": class_lookup.get(cls_id, f"class_{cls_id}"),
                    "bbox_xyxy": [x1, y1, x2, y2],
                })

        return boxes

    def _count_matches(self, gt_boxes: List[Dict[str, Any]], pred_boxes: List[Dict[str, Any]], iou_thr: float) -> int:
        """Matches predictions with ground truth boxes using Intersection over Union (IoU)."""
        matched = 0
        used_pred: set[int] = set()
        
        for gt in gt_boxes:
            gt_cls = int(gt.get("class_id", -1))
            gt_box = gt.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])
            best_j, best_iou = -1, iou_thr
            
            for j, pred in enumerate(pred_boxes):
                if j in used_pred or int(pred.get("class_id", -2)) != gt_cls:
                    continue
                    
                iou = self._iou_xyxy(gt_box, pred.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0]))
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
                    
            if best_j >= 0:
                used_pred.add(best_j)
                matched += 1
                
        return matched

    @staticmethod
    def _iou_xyxy(a: List[float], b: List[float]) -> float:
        """Computes the Intersection over Union metric between two bounding boxes."""
        ax1, ay1, ax2, ay2 = [float(v) for v in a]
        bx1, by1, bx2, by2 = [float(v) for v in b]
        
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = (max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)) + (max(0.0, bx2 - bx1) * max(0.0, by2 - by1)) - inter
        
        if union <= 1e-8:
            return 0.0
        return inter / union

    def _load_class_names(self, dataset_yaml: Path) -> Dict[int, str]:
        """Parses the YAML configuration to extract human-readable class names."""
        names: Dict[int, str] = {}
        if yaml is not None:
            try:
                with open(dataset_yaml, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                raw_names = data.get("names", {})
                if isinstance(raw_names, dict):
                    for k, v in raw_names.items():
                        names[int(k)] = str(v)
                elif isinstance(raw_names, list):
                    for i, v in enumerate(raw_names):
                        names[i] = str(v)
            except Exception:
                names = {}

        if names:
            return names

        in_names = False
        try:
            with open(dataset_yaml, "r", encoding="utf-8") as f:
                for raw in f:
                    stripped = raw.strip()
                    if not stripped: continue
                    
                    if stripped.startswith("names:"):
                        in_names = True
                        continue
                        
                    if in_names and ":" in stripped:
                        left, right = stripped.split(":", 1)
                        left = left.strip()
                        if left.isdigit():
                            names[int(left)] = right.strip().strip('"').strip("'")
        except OSError:
            pass

        return names

    def _dataset_counts(self, dataset_dir: Path) -> tuple[int, int]:
        img_dir = dataset_dir / "images"
        lbl_dir = dataset_dir / "labels"
        img_count = len(self._iter_images(img_dir)) if img_dir.exists() else 0
        lbl_count = len(list(lbl_dir.rglob("*.txt"))) if lbl_dir.exists() else 0
        return img_count, lbl_count

    def _resolve_task(self, dataset_dir: Path) -> str:
        requested = str(getattr(self.config, "task", "auto")).strip().lower()
        if requested in {"detect", "segment"}:
            if requested == "segment" and not self._has_label_files(dataset_dir / "labels"):
                raise ValueError("Segmentation task requires labels in 'labels' directory.")
            return requested

        if self._labels_look_like_segmentation(dataset_dir / "labels") or self._labels_look_like_segmentation(dataset_dir / "labels_seg"):
            return "segment"
        return "detect"

    def _resolve_model_type(self, task: str) -> str:
        if self.config.model_type:
            custom_model = str(self.config.model_type)
            if task == "detect" and ("-seg" in custom_model or custom_model.endswith("seg.pt")):
                raise ValueError(f"Model '{custom_model}' is segmentation-oriented but task is detect.")
            return custom_model
        return "yolov8n-seg.pt" if task == "segment" else "yolov8n.pt"

    def _prepare_segment_variant(self, dataset_dir: Path, safe_name: str) -> tuple[Path, Path]:
        """Prepares a dedicated directory structure for segmentation tasks to isolate data formats recursively."""
        dataset_dir = Path(dataset_dir)
        source_images = dataset_dir / "images"
        source_labels = dataset_dir / "labels"
        source_yaml = dataset_dir / "dataset.yaml"

        if not source_images.exists() or not source_labels.exists() or not source_yaml.exists():
            raise FileNotFoundError(f"Missing required files for segmentation preparation in: {dataset_dir}")

        labels_seg = dataset_dir / "labels_seg"
        if not self._labels_look_like_segmentation(labels_seg):
            self._build_labels_seg_from_detection(source_labels, labels_seg)

        if not self._labels_look_like_segmentation(labels_seg):
            raise ValueError("Cannot run segmentation benchmark: labels are detection format and conversion failed.")

        prepared_dir = self.output_dir / "_prepared_segment" / safe_name
        if prepared_dir.exists():
            shutil.rmtree(prepared_dir, ignore_errors=True)

        prepared_images = prepared_dir / "images"
        prepared_labels = prepared_dir / "labels"
        prepared_images.mkdir(parents=True, exist_ok=True)
        prepared_labels.mkdir(parents=True, exist_ok=True)

        for image_file in self._iter_images(source_images):
            rel = image_file.relative_to(source_images)
            target_img = prepared_images / rel
            target_img.parent.mkdir(parents=True, exist_ok=True)
            self._link_or_copy(image_file, target_img)

        for label_file in sorted(labels_seg.rglob("*.txt")):
            rel = label_file.relative_to(labels_seg)
            target_lbl = prepared_labels / rel
            target_lbl.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(label_file, target_lbl)

        prepared_yaml = prepared_dir / "dataset.yaml"
        self._write_prepared_dataset_yaml(source_yaml, prepared_yaml, prepared_dir)
        return prepared_dir, prepared_yaml

    def _build_labels_seg_from_detection(self, source_labels: Path, target_labels_seg: Path) -> None:
        """Converts bounding box annotations into generic rectangular polygons for segmentation fallbacks recursively."""
        target_labels_seg.mkdir(parents=True, exist_ok=True)

        for det_file in sorted(source_labels.rglob("*.txt")):
            rel_path = det_file.relative_to(source_labels)
            seg_file = target_labels_seg / rel_path
            seg_file.parent.mkdir(parents=True, exist_ok=True)
            seg_lines: List[str] = []

            try:
                with open(det_file, "r", encoding="utf-8") as f:
                    for raw_line in f:
                        line = raw_line.strip()
                        if not line: continue

                        parts = line.split()
                        if len(parts) == 5:
                            cls_id = parts[0]
                            try:
                                cx, cy = float(parts[1]), float(parts[2])
                                bw, bh = float(parts[3]), float(parts[4])
                            except ValueError:
                                continue

                            x1 = max(0.0, min(1.0, cx - bw / 2.0))
                            y1 = max(0.0, min(1.0, cy - bh / 2.0))
                            x2 = max(0.0, min(1.0, cx + bw / 2.0))
                            y2 = max(0.0, min(1.0, cy + bh / 2.0))
                            
                            if x2 <= x1 or y2 <= y1: continue

                            seg_lines.append(f"{cls_id} {x1:.6f} {y1:.6f} {x2:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x1:.6f} {y2:.6f}")
                        elif len(parts) > 5:
                            seg_lines.append(line)
            except OSError:
                seg_lines = []

            with open(seg_file, "w", encoding="utf-8") as f:
                f.write("\n".join(seg_lines) + "\n" if seg_lines else "")

    def _write_prepared_dataset_yaml(self, source_yaml: Path, target_yaml: Path, prepared_dir: Path) -> None:
        try:
            text = source_yaml.read_text(encoding="utf-8")
        except OSError:
            text = ""

        out_lines: List[str] = []
        replaced = {"path": False, "train": False, "val": False, "test": False}

        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("path:"):
                out_lines.append(f"path: {prepared_dir.as_posix()}")
                replaced["path"] = True
            elif stripped.startswith("train:"):
                out_lines.append("train: images/train")
                replaced["train"] = True
            elif stripped.startswith("val:"):
                out_lines.append("val: images/val")
                replaced["val"] = True
            elif stripped.startswith("test:"):
                out_lines.append("test: images/test")
                replaced["test"] = True
            else:
                out_lines.append(line)

        if not replaced["path"]: out_lines.insert(0, f"path: {prepared_dir.as_posix()}")
        if not replaced["train"]: out_lines.append("train: images/train")
        if not replaced["val"]: out_lines.append("val: images/val")
        # Ensure test split is kept if it exists in the new images folder
        if not replaced["test"] and (prepared_dir / "images" / "test").exists():
            out_lines.append("test: images/test")

        target_yaml.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    @staticmethod
    def _iter_images(image_dir: Path) -> List[Path]:
        files: List[Path] = []
        for pattern in ("*.jpg", "*.jpeg", "*.png"):
            files.extend(sorted(image_dir.rglob(pattern)))
        return files

    @staticmethod
    def _link_or_copy(source: Path, target: Path) -> None:
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)

    @staticmethod
    def _labels_look_like_segmentation(label_dir: Path) -> bool:
        if not label_dir.exists(): return False
        for label_file in sorted(label_dir.rglob("*.txt")):
            try:
                with open(label_file, "r", encoding="utf-8") as f:
                    for line in f:
                        stripped = line.strip()
                        if not stripped: continue
                        return len(stripped.split()) > 5
            except OSError:
                continue
        return False

    @staticmethod
    def _has_label_files(label_dir: Path) -> bool:
        return any(label_dir.rglob("*.txt")) if label_dir.exists() else False

    @staticmethod
    def _safe_name(name: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(name))
        return safe.strip("_") or "variant"