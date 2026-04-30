from enum import Enum, auto
from typing import Optional
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from image_model import ImageModel
from selection_model import SelectionModel, SelectionType
from sam2_service import SAM2Service
from sr_service import SRService
from image_operations import OperationRegistry


class AppMode(Enum):
    VIEW = auto()
    SAM2_POSITIVE = auto()
    SAM2_NEGATIVE = auto()
    SAM2_BBOX = auto()
    CROP_BBOX = auto()


class AppController(QObject):
    mode_changed = Signal(object)
    image_loaded = Signal()
    image_changed = Signal()
    selection_changed = Signal()
    prompts_changed = Signal()
    sam2_loading = Signal(str)
    sam2_loaded = Signal()
    sam2_error = Signal(str)
    sr_loading = Signal(str)
    sr_loaded = Signal()
    sr_error = Signal(str)
    status_message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image_model = ImageModel()
        self._selection_model = SelectionModel()
        self._sam2_service = SAM2Service()
        self._sr_service = SRService()
        self._current_mode = AppMode.SAM2_POSITIVE
        self._multimask = True

        # SAM2 wiring
        self._sam2_service.load_progress.connect(self.sam2_loading)
        self._sam2_service.model_loaded.connect(self._on_sam2_loaded)
        self._sam2_service.model_load_failed.connect(self.sam2_error)
        self._sam2_service.prediction_complete.connect(self._on_prediction_complete)
        self._sam2_service.prediction_failed.connect(self._on_prediction_failed)

        # SR wiring
        self._sr_service.load_progress.connect(self.sr_loading)
        self._sr_service.model_loaded.connect(self._on_sr_loaded)
        self._sr_service.model_load_failed.connect(self._on_sr_error)
        self._sr_service.upscale_progress.connect(self.status_message)
        self._sr_service.upscale_complete.connect(self._on_sr_complete)
        self._sr_service.upscale_failed.connect(self._on_sr_error)

        self._selection_model.selection_changed.connect(self.selection_changed)
        self._selection_model.prompts_changed.connect(self.prompts_changed)

    @property
    def mode(self) -> AppMode:
        return self._current_mode

    @property
    def image_model(self) -> ImageModel:
        return self._image_model

    @property
    def selection_model(self) -> SelectionModel:
        return self._selection_model

    @property
    def sr_service(self) -> SRService:
        return self._sr_service

    @property
    def multimask(self) -> bool:
        return self._multimask

    @multimask.setter
    def multimask(self, value: bool):
        self._multimask = value

    def set_mode(self, mode: AppMode):
        self._current_mode = mode
        self.mode_changed.emit(mode)
        names = {
            AppMode.SAM2_POSITIVE: "Positive Point",
            AppMode.SAM2_NEGATIVE: "Negative Point",
            AppMode.SAM2_BBOX: "SAM2 BBox",
            AppMode.CROP_BBOX: "Crop BBox",
        }
        p = self._selection_model.prompts
        self.status_message.emit(
            f"Mode: {names.get(mode, 'Unknown')} | "
            f"Pos:{len(p.positive_points)} Neg:{len(p.negative_points)} "
            f"BBox:{'Y' if p.bbox else 'N'}"
        )

    def load_image(self, path: str) -> bool:
        try:
            if not self._image_model.load_image(path):
                self.sam2_error.emit(f"Cannot load image: {path}")
                return False
            self._selection_model.clear_selection()
            self._sam2_service.reset_image()
            self.image_loaded.emit()
            self.status_message.emit(f"Loaded: {Path(path).name}")
            self._sam2_service.set_image(self._image_model.current_image_rgb)
            return True
        except Exception as e:
            self.sam2_error.emit(f"Failed to load image: {str(e)}")
            return False

    def load_sam2_model(self, config_path: str, checkpoint_path: str):
        self.sam2_loading.emit("Loading SAM2 model...")
        self._sam2_service.load_model(config_path, checkpoint_path)

    def _on_sam2_loaded(self):
        self.sam2_loaded.emit()
        self.status_message.emit("SAM2 model ready")

    def load_sr_model(self):
        self._sr_service.load_model(scale=4)

    def _on_sr_loaded(self):
        self.sr_loaded.emit()
        self.status_message.emit("SR model ready")

    def _on_sr_error(self, error: str):
        self.sr_error.emit(error)

    # -- Prompt accumulation (no auto-predict) --

    def handle_canvas_click(self, x: int, y: int):
        if self._current_mode == AppMode.SAM2_POSITIVE:
            self._selection_model.add_positive_point(x, y)
        elif self._current_mode == AppMode.SAM2_NEGATIVE:
            self._selection_model.add_negative_point(x, y)

    def handle_bbox_drag(self, x0: int, y0: int, x1: int, y1: int):
        bbox = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        if self._current_mode == AppMode.SAM2_BBOX:
            self._selection_model.set_bbox_prompt(bbox)
        elif self._current_mode == AppMode.CROP_BBOX:
            self._selection_model.set_manual_bbox(bbox)
            self.status_message.emit(f"Crop region: {bbox[2]-bbox[0]}x{bbox[3]-bbox[1]}")

    def confirm_prompts(self):
        """Run SAM2 prediction with all accumulated prompts."""
        prompts = self._selection_model.prompts
        if not prompts.positive_points and not prompts.negative_points and not prompts.bbox:
            self.status_message.emit("No prompts! Add points or bbox first.")
            return

        if not self._sam2_service.is_loaded:
            self.sam2_error.emit("SAM2 model not loaded yet")
            return

        self.status_message.emit("Running SAM2 prediction...")
        self._sam2_service.predict(
            positive_points=prompts.positive_points,
            negative_points=prompts.negative_points,
            bbox=prompts.bbox,
            multimask=self._multimask
        )

    def _on_prediction_complete(self, masks: np.ndarray, scores: np.ndarray):
        self._selection_model.set_sam2_mask(masks, scores)
        self.status_message.emit(
            f"Prediction done. Best score: {scores.max():.3f} "
            f"({self._selection_model.num_masks} masks). "
            f"Keys 1/2/3 to cycle masks."
        )

    def _on_prediction_failed(self, error: str):
        self.sam2_error.emit(error)

    def cycle_sam2_mask(self, idx: int):
        if self._selection_model.num_masks == 0:
            return
        n = self._selection_model.num_masks
        if idx < n:
            self._selection_model._selected_idx = idx
            self._selection_model.mask_changed.emit()
            self._selection_model.selection_changed.emit()
            scores = self._selection_model._sam2_scores
            self.status_message.emit(
                f"Mask {idx+1}/{n} score: {scores[idx]:.3f}"
            )

    def undo_last_prompt(self):
        prompts = self._selection_model.prompts
        undone = False
        if self._current_mode == AppMode.SAM2_POSITIVE:
            if prompts.positive_points:
                prompts.positive_points.pop()
                undone = True
        elif self._current_mode == AppMode.SAM2_NEGATIVE:
            if prompts.negative_points:
                prompts.negative_points.pop()
                undone = True
        elif self._current_mode == AppMode.SAM2_BBOX:
            if prompts.bbox is not None:
                prompts.bbox = None
                undone = True
        if undone:
            self._selection_model.prompts_changed.emit()

    def clear_all(self):
        self._selection_model.clear_selection()
        self.status_message.emit("Cleared all prompts and selection")

    # -- Super Resolution --

    def apply_super_resolution(self, params: dict):
        if not self._image_model.is_loaded:
            self.status_message.emit("No image loaded")
            return
        if self._sr_service.is_upscaling:
            self.status_message.emit("SR already running")
            return
        if not self._sr_service.is_loaded:
            self.status_message.emit("SR model not loaded yet")
            return

        self.status_message.emit("SR: Starting super resolution...")
        self._sr_service.upscale(self._image_model.current_image_rgb, params)

    def _on_sr_complete(self, result: np.ndarray):
        if result.dtype != np.uint8:
            result = np.clip(result, 0, 255).astype(np.uint8)
        if result.ndim == 3 and result.shape[2] == 4:
            result = result[:, :, :3]
        self._image_model.set_image(result)
        self.image_changed.emit()
        h, w = result.shape[:2]
        self.status_message.emit(f"SR: Complete {w}x{h}")

    # -- Image operations --

    def apply_operation(self, operation_id: str, **kwargs):
        if not self._selection_model.has_selection:
            self.status_message.emit("No selection to apply operation")
            return False

        operation = OperationRegistry.get(operation_id)
        if operation is None:
            self.status_message.emit(f"Unknown operation: {operation_id}")
            return False

        for key, value in kwargs.items():
            if hasattr(operation, key):
                setattr(operation, key, value)

        mask = self._selection_model.get_mask_for_operation(
            self._image_model.current_image_rgb.shape
        )
        result = operation.apply(self._image_model.current_image_rgb, mask)
        self._image_model.set_image(result)
        self.image_changed.emit()
        self.status_message.emit(f"Applied {operation.name}")
        return True

    def apply_mosaic(self, block_size: int = 10):
        return self.apply_operation("mosaic", block_size=block_size)

    def get_crop_region(self) -> Optional[np.ndarray]:
        bounds = self._selection_model.get_selection_bounds()
        if bounds is None:
            return None
        x0, y0, x1, y1 = bounds
        img = self._image_model.current_image_rgb
        return img[y0:y1 + 1, x0:x1 + 1]

    def save_selection(self, path: str) -> bool:
        region = self.get_crop_region()
        if region is None:
            self.status_message.emit("No selection to save")
            return False
        try:
            from PIL import Image
            Image.fromarray(region).save(path)
            self.status_message.emit(f"Saved: {path}")
            return True
        except Exception as e:
            self.status_message.emit(f"Save failed: {str(e)}")
            return False