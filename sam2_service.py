from typing import Optional, List, Tuple

import numpy as np
import torch
from PySide6.QtCore import QObject, Signal, QThread


class SAM2LoadWorker(QThread):
    finished = Signal(object)
    progress = Signal(str)
    error = Signal(str)

    def __init__(self, config_path: str, checkpoint_path: str, parent=None):
        super().__init__(parent)
        self._config_path = config_path
        self._checkpoint_path = checkpoint_path

    def run(self):
        try:
            self.progress.emit("Determining device...")
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

            self.progress.emit(f"Loading SAM2 model on {device}...")
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            model = build_sam2(self._config_path, self._checkpoint_path, device=device)
            predictor = SAM2ImagePredictor(model)

            self.progress.emit("Model loaded successfully")
            self.finished.emit(predictor)
        except Exception as e:
            self.error.emit(str(e))


class SAM2Service(QObject):
    model_loaded = Signal()
    model_load_failed = Signal(str)
    load_progress = Signal(str)
    image_set = Signal()
    prediction_complete = Signal(np.ndarray, np.ndarray)
    prediction_failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._predictor = None
        self._is_loading = False
        self._is_loaded = False
        self._current_image: Optional[np.ndarray] = None

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def is_loading(self) -> bool:
        return self._is_loading

    def load_model(self, config_path: str, checkpoint_path: str):
        if self._is_loading or self._is_loaded:
            return
        self._is_loading = True
        self._worker = SAM2LoadWorker(config_path, checkpoint_path)
        self._worker.progress.connect(self.load_progress)
        self._worker.finished.connect(self._on_model_loaded)
        self._worker.error.connect(self._on_model_error)
        self._worker.start()

    def _on_model_loaded(self, predictor):
        self._predictor = predictor
        self._is_loaded = True
        self._is_loading = False
        self.model_loaded.emit()
        if self._current_image is not None:
            self.set_image(self._current_image)

    def _on_model_error(self, error: str):
        self._is_loading = False
        self.model_load_failed.emit(error)

    def set_image(self, image: np.ndarray):
        self._current_image = image
        if self._predictor is None:
            return
        try:
            self._predictor.set_image(image)
            self.image_set.emit()
        except Exception as e:
            self.prediction_failed.emit(f"Failed to set image: {str(e)}")

    def predict(
        self,
        positive_points: List[Tuple[int, int]],
        negative_points: List[Tuple[int, int]],
        bbox: Optional[Tuple[int, int, int, int]] = None,
        multimask: bool = True
    ):
        if self._predictor is None:
            self.prediction_failed.emit("Model not loaded")
            return
        if self._current_image is None:
            self.prediction_failed.emit("No image set")
            return

        try:
            point_coords = []
            point_labels = []

            for x, y in positive_points:
                point_coords.append([x, y])
                point_labels.append(1)
            for x, y in negative_points:
                point_coords.append([x, y])
                point_labels.append(0)

            point_coords_arr = np.array(point_coords, dtype=np.float32) if point_coords else None
            point_labels_arr = np.array(point_labels, dtype=np.int32) if point_labels else None
            bbox_arr = np.array(bbox, dtype=np.float32)[None, :] if bbox else None

            if point_coords_arr is None and bbox_arr is None:
                self.prediction_failed.emit("No prompts provided")
                return

            masks, scores, _ = self._predictor.predict(
                point_coords=point_coords_arr,
                point_labels=point_labels_arr,
                box=bbox_arr,
                multimask_output=multimask
            )
            self.prediction_complete.emit(masks, scores)
        except Exception as e:
            self.prediction_failed.emit(f"Prediction failed: {str(e)}")

    def reset_image(self):
        self._current_image = None