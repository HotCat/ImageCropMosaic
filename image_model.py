from typing import Optional
from pathlib import Path
import numpy as np
from PIL import Image


class ImageModel:
    def __init__(self):
        self._original: Optional[np.ndarray] = None
        self._current: Optional[np.ndarray] = None
        self._path: Optional[str] = None

    @property
    def current_image_rgb(self) -> Optional[np.ndarray]:
        return self._current

    @property
    def image_path(self) -> Optional[str]:
        return self._path

    @property
    def is_loaded(self) -> bool:
        return self._current is not None

    def load_image(self, path: str) -> bool:
        p = Path(path)
        if not p.exists():
            return False
        img = Image.open(p).convert("RGB")
        self._original = np.array(img)
        self._current = self._original.copy()
        self._path = str(p)
        return True

    def set_image(self, image: np.ndarray):
        self._current = image.copy()

    def reset_to_original(self):
        if self._original is not None:
            self._current = self._original.copy()
