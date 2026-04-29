from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import numpy as np
from PySide6.QtCore import QObject, Signal


class SelectionType(Enum):
    NONE = auto()
    SAM2_MASK = auto()
    MANUAL_BBOX = auto()


@dataclass
class SAM2Prompt:
    positive_points: List[Tuple[int, int]] = field(default_factory=list)
    negative_points: List[Tuple[int, int]] = field(default_factory=list)
    bbox: Optional[Tuple[int, int, int, int]] = None


class SelectionModel(QObject):
    selection_changed = Signal()
    mask_changed = Signal()
    bbox_changed = Signal()
    prompts_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._selection_type = SelectionType.NONE
        self._sam2_masks: Optional[np.ndarray] = None
        self._sam2_scores: Optional[np.ndarray] = None
        self._selected_idx: int = 0
        self._manual_bbox: Optional[Tuple[int, int, int, int]] = None
        self._prompts = SAM2Prompt()

    @property
    def selection_type(self) -> SelectionType:
        return self._selection_type

    @property
    def has_selection(self) -> bool:
        return self._selection_type != SelectionType.NONE

    @property
    def selected_mask(self) -> Optional[np.ndarray]:
        if self._selection_type == SelectionType.SAM2_MASK and self._sam2_masks is not None:
            if self._sam2_masks.ndim == 3:
                return self._sam2_masks[self._selected_idx]
            return self._sam2_masks
        return None

    @property
    def manual_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        if self._selection_type == SelectionType.MANUAL_BBOX:
            return self._manual_bbox
        return None

    @property
    def prompts(self) -> SAM2Prompt:
        return self._prompts

    @property
    def num_masks(self) -> int:
        if self._sam2_masks is not None and self._sam2_masks.ndim == 3:
            return self._sam2_masks.shape[0]
        return 0

    def get_selection_bounds(self) -> Optional[Tuple[int, int, int, int]]:
        if self._selection_type == SelectionType.MANUAL_BBOX:
            return self._manual_bbox
        elif self._selection_type == SelectionType.SAM2_MASK:
            mask = self.selected_mask
            if mask is not None:
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                if rows.any() and cols.any():
                    y0, y1 = np.where(rows)[0][[0, -1]]
                    x0, x1 = np.where(cols)[0][[0, -1]]
                    return (int(x0), int(y0), int(x1), int(y1))
        return None

    def get_mask_for_operation(self, image_shape: Tuple[int, int]) -> np.ndarray:
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        if self._selection_type == SelectionType.SAM2_MASK:
            sam_mask = self.selected_mask
            if sam_mask is not None:
                mask = sam_mask > 0.5
        elif self._selection_type == SelectionType.MANUAL_BBOX:
            if self._manual_bbox is not None:
                x0, y0, x1, y1 = self._manual_bbox
                mask[y0:y1 + 1, x0:x1 + 1] = True
        return mask

    def set_sam2_mask(self, masks: np.ndarray, scores: np.ndarray):
        self._selection_type = SelectionType.SAM2_MASK
        self._sam2_masks = masks
        self._sam2_scores = scores
        self._selected_idx = int(np.argmax(scores))
        self._manual_bbox = None
        self.mask_changed.emit()
        self.selection_changed.emit()

    def set_manual_bbox(self, bbox: Tuple[int, int, int, int]):
        self._selection_type = SelectionType.MANUAL_BBOX
        self._manual_bbox = bbox
        self._sam2_masks = None
        self.mask_changed.emit()
        self.bbox_changed.emit()
        self.selection_changed.emit()

    def add_positive_point(self, x: int, y: int):
        self._prompts.positive_points.append((x, y))
        self.prompts_changed.emit()

    def add_negative_point(self, x: int, y: int):
        self._prompts.negative_points.append((x, y))
        self.prompts_changed.emit()

    def set_bbox_prompt(self, bbox: Optional[Tuple[int, int, int, int]]):
        self._prompts.bbox = bbox
        self.prompts_changed.emit()

    def clear_prompts(self):
        self._prompts = SAM2Prompt()
        self.prompts_changed.emit()

    def clear_selection(self):
        self._selection_type = SelectionType.NONE
        self._sam2_masks = None
        self._sam2_scores = None
        self._selected_idx = 0
        self._manual_bbox = None
        self._prompts = SAM2Prompt()
        self.selection_changed.emit()

    def cycle_mask(self, direction: int = 1):
        if self._selection_type == SelectionType.SAM2_MASK and self._sam2_masks is not None:
            if self._sam2_masks.ndim == 3:
                n = self._sam2_masks.shape[0]
                self._selected_idx = (self._selected_idx + direction) % n
                self.mask_changed.emit()
                self.selection_changed.emit()
