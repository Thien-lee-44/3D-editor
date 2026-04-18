import logging
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Union, Dict, Any
from ultralytics import YOLO

# =========================================================================
# LOGGING CONFIGURATION
# =========================================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("AIPipeline")

# =========================================================================
# CONFIGURATION DATA CLASSES
# =========================================================================
@dataclass
class TrainingConfig:
    """
    Data container for YOLO training hyperparameters.
    Decouples configuration from execution to ensure extensibility.
    """
    epochs: int = 100
    batch_size: int = 16
    imgsz: int = 640
    patience: int = 20
    save_period: int = 10
    cache: bool = True
    workers: int = 4
    
    # Advanced Data Augmentation tuned for Synthetic Datasets
    augmentation: Dict[str, float] = field(default_factory=lambda: {
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "fliplr": 0.5
    })
    
    device: str = field(init=False)

    def __post_init__(self) -> None:
        """Automatically resolves the optimal hardware accelerator upon initialization."""
        self.device = self._detect_hardware()

    @staticmethod
    def _detect_hardware() -> str:
        if torch.cuda.is_available():
            return "0"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

# =========================================================================
# PIPELINE ARCHITECTURE
# =========================================================================
class YOLOPipeline:
    """
    Professional End-to-End AI Training and Deployment Pipeline for YOLO architectures.
    Provides automated dataset validation, optimized training, evaluation, and production export.
    """

    def __init__(self, dataset_yaml: Union[str, Path], model_type: str = "yolov8n.pt", project_name: str = "Synthetic_AI") -> None:
        """
        Initializes the AI pipeline ecosystem.
        
        Args:
            dataset_yaml (Union[str, Path]): Absolute path to the dataset.yaml descriptor.
            model_type (str): Pre-trained model weight identifier (e.g., 'yolov8n.pt').
            project_name (str): Directory namespace for MLflow/runs tracking.
        """
        self.dataset_yaml = Path(dataset_yaml)
        self.project_name = project_name
        self.config = TrainingConfig()
        
        if not self.dataset_yaml.exists():
            logger.error(f"Dataset descriptor not found at: {self.dataset_yaml}")
            raise FileNotFoundError(f"Missing dataset descriptor: {self.dataset_yaml}")

        logger.info(f"Initializing YOLO Model: {model_type} on device: {self.config.device.upper()}")
        self.model = YOLO(model_type)

    def validate_dataset(self) -> bool:
        """
        Performs pre-training integrity checks on the dataset directory structure and contents.
        
        Returns:
            bool: True if the dataset meets structural requirements, False otherwise.
        """
        logger.info("Initiating dataset integrity validation...")
        
        dataset_dir = self.dataset_yaml.parent
        img_dir = dataset_dir / "images"
        lbl_dir = dataset_dir / "labels"

        if not img_dir.exists() or not lbl_dir.exists():
            logger.error("Invalid dataset structure. Missing 'images' or 'labels' directories.")
            return False

        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        labels = list(lbl_dir.glob("*.txt"))

        logger.info(f"Discovered {len(images)} images and {len(labels)} annotation files.")
        
        if len(images) != len(labels):
            logger.warning("Dataset asymmetry detected: Number of images does not match annotations.")
        
        empty_labels = [lbl for lbl in labels if lbl.stat().st_size == 0]
        if empty_labels:
            logger.warning(f"Detected {len(empty_labels)} empty annotation files (Potential occlusion).")

        logger.info("Dataset validation passed successfully.")
        return True

    def train(self, custom_config: Optional[TrainingConfig] = None) -> None:
        """
        Executes the optimization loop based on the injected configuration payload.
        
        Args:
            custom_config (Optional[TrainingConfig]): Override default hyperparameters.
        """
        active_config = custom_config or self.config
        logger.info(f"Starting training phase for {active_config.epochs} epochs...")
        
        try:
            self.model.train(
                data=str(self.dataset_yaml),
                epochs=active_config.epochs,
                batch=active_config.batch_size,
                imgsz=active_config.imgsz,
                device=active_config.device,
                project=self.project_name,
                name="train_run",
                exist_ok=True,
                patience=active_config.patience,
                save_period=active_config.save_period,
                cache=active_config.cache,
                workers=active_config.workers,
                **active_config.augmentation
            )
            logger.info("Training phase completed successfully.")
        except Exception as e:
            logger.exception("An error occurred during the training loop.")
            raise

    def evaluate(self) -> None:
        """Computes evaluation metrics (mAP) against the validation subset."""
        logger.info("Initiating model evaluation pipeline...")
        try:
            metrics = self.model.val(device=self.config.device)
            logger.info(f"Evaluation complete -> mAP@50: {metrics.box.map50:.4f} | mAP@50-95: {metrics.box.map:.4f}")
        except Exception as e:
            logger.exception("Model evaluation failed.")
            raise

    def export_production(self, format_type: str = "onnx") -> Optional[str]:
        """
        Serializes the trained weights into an optimized format for inference engines.
        
        Args:
            format_type (str): Target export format (e.g., 'onnx', 'engine', 'tflite').
            
        Returns:
            Optional[str]: Absolute path to the exported binary.
        """
        logger.info(f"Exporting model to production format: {format_type.upper()}")
        best_model_path = Path(self.project_name) / "train_run" / "weights" / "best.pt"
        
        if not best_model_path.exists():
            logger.error("Missing 'best.pt' weights. Ensure training was completed successfully.")
            return None

        try:
            prod_model = YOLO(str(best_model_path))
            export_path = prod_model.export(format=format_type, dynamic=True)
            logger.info(f"Export successful. Artifact located at: {export_path}")
            return export_path
        except Exception as e:
            logger.exception("Export process failed.")
            return None

    def predict_sample(self, image_path: Union[str, Path], confidence_threshold: float = 0.5) -> None:
        """
        Executes a localized inference pass on a specified image for visual verification.
        
        Args:
            image_path (Union[str, Path]): Target image path.
            confidence_threshold (float): Minimum confidence score to register a detection.
        """
        target_path = Path(image_path)
        if not target_path.exists():
            logger.error(f"Inference target not found: {target_path}")
            return

        logger.info(f"Running inference on target: {target_path.name}")
        best_model_path = Path(self.project_name) / "train_run" / "weights" / "best.pt"
        model_instance = YOLO(str(best_model_path)) if best_model_path.exists() else self.model
        
        try:
            results = model_instance.predict(
                source=str(target_path), 
                save=True, 
                conf=confidence_threshold, 
                show_labels=True, 
                show_conf=True
            )
            logger.info(f"Inference complete. Visualized output saved to: {results[0].save_dir}")
        except Exception as e:
            logger.exception("Inference execution failed.")

# =====================================================================
# EXECUTION ENTRY POINT
# =====================================================================
if __name__ == '__main__':
    # Define absolute path to the generated YAML descriptor
    DATASET_DESCRIPTOR = r"datasets\export_20260418_103000\dataset.yaml"
    
    try:
        # Initialize pipeline ecosystem
        pipeline = YOLOPipeline(
            dataset_yaml=DATASET_DESCRIPTOR, 
            model_type="yolov8n.pt", 
            project_name="Synthetic_Autonomous_Driving"
        )

        if pipeline.validate_dataset():
            # Override default config via Dataclass injection
            custom_cfg = TrainingConfig()
            custom_cfg.epochs = 50
            custom_cfg.batch_size = 16
            
            # Execute Pipeline
            pipeline.train(custom_config=custom_cfg)
            pipeline.evaluate()
            pipeline.export_production(format_type="onnx")
            
            # Validate output visually
            sample_img = Path(DATASET_DESCRIPTOR).parent / "images" / "frame_00000.jpg"
            if sample_img.exists():
                pipeline.predict_sample(sample_img)
                
    except Exception as main_e:
        logger.critical("Pipeline execution terminated due to a fatal error.")