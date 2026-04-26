import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import torch

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


@dataclass
class CVProofConfig:
    model_type: str = "yolov8n-seg.pt"  # Thay đổi thành mô hình Segmentation
    epochs: int = 3
    batch_size: int = 8
    imgsz: int = 640
    confidence_threshold: float = 0.25
    run_training: bool = True


class CVProofRunner:
    """
    Runs reproducible CV proof benchmarks for one or more dataset variants.
    Exports JSON/CSV/Markdown artifacts for report-ready evidence, supporting both Box & Mask metrics.
    """

    def __init__(self, output_dir: Path, config: Optional[CVProofConfig] = None) -> None:
        self.output_dir = Path(output_dir)
        self.config = config or CVProofConfig()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "yolo_runs").mkdir(parents=True, exist_ok=True)

    def run(
        self,
        variant_dirs: Dict[str, Path],
        progress_cb: Optional[Callable[[str, int, int], None]] = None,
    ) -> Dict[str, Any]:
        
        if YOLO is None:
            raise RuntimeError("Ultralytics YOLO is not installed. Please run: pip install ultralytics")

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
                records.append(
                    {
                        "variant": variant_name,
                        "status": "missing_dataset_yaml",
                        "dataset_dir": str(variant_dir),
                    }
                )
                continue

            try:
                project_dir = self.output_dir / "yolo_runs" / safe_name
                
                # 1. Initialize Direct Ultralytics Model
                model = YOLO(self.config.model_type)

                # 2. Hardware-Accelerated Training
                if self.config.run_training:
                    device = "0" if torch.cuda.is_available() else "cpu"
                    workers = 8 if torch.cuda.is_available() else 2
                    
                    model.train(
                        data=str(dataset_yaml),
                        epochs=int(self.config.epochs),
                        batch=int(self.config.batch_size),
                        imgsz=int(self.config.imgsz),
                        project=str(project_dir),
                        name="train",
                        exist_ok=True,
                        device=device,
                        workers=workers,
                        patience=5, # Early stopping chống Overfitting
                        verbose=False,
                        plots=False
                    )
                    
                    best_weights = project_dir / "train" / "weights" / "best.pt"
                    if best_weights.exists():
                        model = YOLO(str(best_weights))

                # 3. Evaluation & Dual-Metric Extraction (Box + Seg)
                metrics_out = model.val(
                    data=str(dataset_yaml),
                    project=str(project_dir),
                    name="val",
                    exist_ok=True,
                    verbose=False,
                    split="val"
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

                # 4. Sample Prediction
                sample_img = self._first_image(variant_dir / "images")
                pred_dir = None
                if sample_img:
                    model.predict(
                        source=str(sample_img),
                        conf=float(self.config.confidence_threshold),
                        project=str(project_dir),
                        name="predict",
                        save=True,
                        exist_ok=True,
                        verbose=False
                    )
                    pred_dir = str(project_dir / "predict")

                img_count, lbl_count = self._dataset_counts(variant_dir)

                records.append(
                    {
                        "variant": variant_name,
                        "status": "ok",
                        "dataset_dir": str(variant_dir),
                        "dataset_yaml": str(dataset_yaml),
                        "num_images": img_count,
                        "num_labels": lbl_count,
                        "metrics": metrics,
                        "prediction_dir": pred_dir,
                    }
                )
            except Exception as exc:
                records.append(
                    {
                        "variant": variant_name,
                        "status": "error",
                        "dataset_dir": str(variant_dir),
                        "error": str(exc),
                    }
                )

        artifacts = self._write_artifacts(records)
        return {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "records": records,
            "artifacts": artifacts,
        }

    def _write_artifacts(self, records: List[Dict[str, Any]]) -> Dict[str, str]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_json = self.output_dir / f"cv_proof_metrics_{timestamp}.json"
        metrics_csv = self.output_dir / f"cv_proof_metrics_{timestamp}.csv"
        summary_md = self.output_dir / f"cv_proof_summary_{timestamp}.md"

        with open(metrics_json, "w", encoding="utf-8") as f:
            json.dump({"records": records}, f, ensure_ascii=False, indent=2)

        csv_columns = [
            "variant", "status", "dataset_dir", "num_images", "num_labels",
            "box_map50", "box_map50_95", "box_precision", "box_recall",
            "seg_map50", "seg_map50_95", "seg_precision", "seg_recall",
            "prediction_dir", "error",
        ]
        with open(metrics_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writeheader()

            for rec in records:
                m = rec.get("metrics", {})
                writer.writerow(
                    {
                        "variant": rec.get("variant", ""),
                        "status": rec.get("status", ""),
                        "dataset_dir": rec.get("dataset_dir", ""),
                        "num_images": rec.get("num_images", 0),
                        "num_labels": rec.get("num_labels", 0),
                        "box_map50": float(m.get("box_map50", 0.0)) if m else 0.0,
                        "box_map50_95": float(m.get("box_map50_95", 0.0)) if m else 0.0,
                        "box_precision": float(m.get("box_precision", 0.0)) if m else 0.0,
                        "box_recall": float(m.get("box_recall", 0.0)) if m else 0.0,
                        "seg_map50": float(m.get("seg_map50", 0.0)) if m else 0.0,
                        "seg_map50_95": float(m.get("seg_map50_95", 0.0)) if m else 0.0,
                        "seg_precision": float(m.get("seg_precision", 0.0)) if m else 0.0,
                        "seg_recall": float(m.get("seg_recall", 0.0)) if m else 0.0,
                        "prediction_dir": rec.get("prediction_dir", ""),
                        "error": rec.get("error", ""),
                    }
                )

        with open(summary_md, "w", encoding="utf-8") as f:
            f.write("# CV Proof Summary\n\n")
            f.write(f"Generated at: {datetime.now().isoformat(timespec='seconds')}\n\n")

            ok_records = [r for r in records if r.get("status") == "ok"]
            # Ranking based on Mask Segmentation mAP@50 (The Bonus Requirement)
            ranked = sorted(ok_records, key=lambda r: float(r.get("metrics", {}).get("seg_map50", 0.0)), reverse=True)

            if not ranked:
                f.write("No successful benchmark run was produced.\n")
            else:
                f.write("## Ranking by Segmentation mAP@50 (Mask Accuracy)\n\n")
                for i, rec in enumerate(ranked, start=1):
                    m = rec.get("metrics", {})
                    f.write(
                        f"{i}. **{rec.get('variant', 'Unknown')}** - "
                        f"Seg mAP@50: **{float(m.get('seg_map50', 0.0)):.4f}** | "
                        f"Box mAP@50: {float(m.get('box_map50', 0.0)):.4f}\n"
                    )

                f.write("\n## Detail Performance Table\n\n")
                f.write("| Variant | Imgs | Lbls | Box mAP@50 | Seg mAP@50 | Box P | Box R | Seg P | Seg R |\n")
                f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
                for rec in ranked:
                    m = rec.get("metrics", {})
                    f.write(
                        f"| {rec.get('variant', 'Unknown')} "
                        f"| {int(rec.get('num_images', 0))} "
                        f"| {int(rec.get('num_labels', 0))} "
                        f"| {float(m.get('box_map50', 0.0)):.4f} "
                        f"| **{float(m.get('seg_map50', 0.0)):.4f}** "
                        f"| {float(m.get('box_precision', 0.0)):.4f} "
                        f"| {float(m.get('box_recall', 0.0)):.4f} "
                        f"| {float(m.get('seg_precision', 0.0)):.4f} "
                        f"| {float(m.get('seg_recall', 0.0)):.4f} |\n"
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

    def _first_image(self, image_dir: Path) -> Optional[Path]:
        if not image_dir.exists():
            return None

        for ext in ("*.jpg", "*.jpeg", "*.png"):
            candidates = sorted(image_dir.glob(ext))
            if candidates:
                return candidates[0]

        return None

    def _dataset_counts(self, dataset_dir: Path) -> tuple[int, int]:
        img_dir = dataset_dir / "images"
        lbl_dir = dataset_dir / "labels"

        img_count = 0
        lbl_count = 0

        if img_dir.exists():
            img_count = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.jpeg"))) + len(list(img_dir.glob("*.png")))
        if lbl_dir.exists():
            lbl_count = len(list(lbl_dir.glob("*.txt")))

        return img_count, lbl_count

    def _safe_name(self, name: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(name))
        return safe.strip("_") or "variant"