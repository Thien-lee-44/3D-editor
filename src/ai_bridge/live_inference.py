import logging
import numpy as np
from typing import List, Tuple, Optional
from ultralytics import YOLO

logger = logging.getLogger("LiveAIEngine")

class LiveAIEngine:
    """
    Real-time bridge between the OpenGL Render Pipeline and the YOLO Inference Engine.
    Processes raw FBO pixel byte arrays directly in memory to achieve zero-I/O latency.
    """
    def __init__(self, default_model_path: str = "yolov8n.pt") -> None:
        self.model_path: str = default_model_path
        self.model: Optional[YOLO] = None
        self.is_active: bool = False
        self.labels: dict = {}

    def toggle(self) -> bool:
        """Toggles the live inference state and lazy-loads the model if necessary."""
        self.is_active = not self.is_active
        if self.is_active and self.model is None:
            self._load_model()
        return self.is_active

    def _load_model(self) -> None:
        """Initializes the tensor graphs and loads weights into VRAM/RAM."""
        try:
            logger.info(f"Loading Live AI Model from {self.model_path}...")
            self.model = YOLO(self.model_path)
            self.labels = self.model.names
            logger.info("Live AI Model successfully loaded and ready for inference.")
        except Exception as e:
            logger.error(f"Failed to load AI Model: {str(e)}")
            self.is_active = False

    def predict_from_fbo(self, rgb_pixels: bytes, width: int, height: int) -> List[Tuple[int, str, float, float, float, float, float]]:
        """
        Translates raw OpenGL framebuffer bytes into tensor structures for YOLO inference.
        
        Returns:
            List of predictions: [(ClassID, Label, Confidence, X_min, Y_min, X_max, Y_max), ...]
        """
        if not rgb_pixels or not self.is_active or self.model is None:
            return []

        try:
            # Convert raw GL bytes to shape (H, W, 3)
            image_array = np.frombuffer(rgb_pixels, dtype=np.uint8).reshape((height, width, 3))
            
            # OpenGL coordinates are bottom-up; ML models expect top-down
            image_array = np.flipud(image_array)

            # Execute inference pass (verbose=False to prevent console flooding)
            results = self.model(image_array, verbose=False)
            
            predictions = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    class_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    
                    label = self.labels.get(class_id, "Unknown")
                    predictions.append((class_id, label, conf, x1, y1, x2, y2))
                    
            return predictions
            
        except Exception as e:
            logger.error(f"Live Inference Exception: {str(e)}")
            return []