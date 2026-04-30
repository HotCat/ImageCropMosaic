import os
import time
from typing import Optional
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from PySide6.QtCore import QObject, Signal, QThread

from config import SR_MODEL_DIR


class SRLoadWorker(QThread):
    finished = Signal(object)
    progress = Signal(str)
    error = Signal(str)

    def __init__(self, scale: int = 4, parent=None):
        super().__init__(parent)
        self._scale = scale

    def run(self):
        try:
            self.progress.emit("Initializing Real-ESRGAN...")

            # Ensure model directory exists
            SR_MODEL_DIR.mkdir(parents=True, exist_ok=True)

            # Download model if not present
            if self._scale == 4:
                model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
                model_name = "RealESRGAN_x4plus.pth"
            else:
                model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
                model_name = "RealESRGAN_x2plus.pth"

            model_path = SR_MODEL_DIR / model_name
            clean_path = SR_MODEL_DIR / f"{model_name[:-4]}_clean.pth"

            if not clean_path.exists():
                if not model_path.exists():
                    self.progress.emit(f"Downloading {model_name} (~64MB)...")
                    import urllib.request
                    urllib.request.urlretrieve(model_url, str(model_path))

                self.progress.emit("Loading model weights...")
                state = torch.load(model_path, map_location='cpu')
                params = state.get('params_ema', state)
                torch.save(params, clean_path)
                model_path.unlink(missing_ok=True)  # Remove original to save space

            self.progress.emit("Building model...")
            from spandrel import ModelLoader
            loader = ModelLoader()
            model = loader.load_from_file(str(clean_path))

            self.progress.emit(f"Real-ESRGAN {self._scale}x ready")
            self.finished.emit((model, self._scale))

        except Exception as e:
            self.error.emit(str(e))


class SRUpscaleWorker(QThread):
    finished = Signal(object)
    progress = Signal(str)
    error = Signal(str)

    def __init__(self, model, scale: int, image_rgb: np.ndarray, tile_size: int = 512, parent=None):
        super().__init__(parent)
        self._model = model
        self._scale = scale
        self._image_rgb = image_rgb
        self._tile_size = tile_size

    def run(self):
        try:
            t0 = time.perf_counter()
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

            h, w = self._image_rgb.shape[:2]
            new_h, new_w = h * self._scale, w * self._scale

            self.progress.emit(f"Upscaling {w}x{h} -> {new_w}x{new_h}...")

            # For small images, process directly
            if max(w, h) <= self._tile_size:
                result = self._upscale_direct(device)
            else:
                result = self._upscale_tiled(device)

            elapsed = time.perf_counter() - t0
            self.finished.emit((result, new_w, new_h, elapsed))

        except Exception as e:
            self.error.emit(str(e))

    def _upscale_direct(self, device) -> np.ndarray:
        """Direct upscale for small images."""
        self._model.model.to(device)
        self._model.model.eval()

        img_tensor = torch.from_numpy(self._image_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

        with torch.no_grad():
            output = self._model.model(img_tensor)

        result = (output.squeeze().permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        return result

    def _upscale_tiled(self, device) -> np.ndarray:
        """Tile-based upscale for large images to avoid OOM."""
        self._model.model.to(device)
        self._model.model.eval()

        h, w = self._image_rgb.shape[:2]
        tile = self._tile_size
        scale = self._scale

        # Calculate output size
        out_h, out_w = h * scale, w * scale
        result = np.zeros((out_h, out_w, 3), dtype=np.uint8)

        # Calculate tile positions with overlap
        overlap = tile // 8  # Small overlap to avoid seams

        tiles_x = max(1, (w - overlap) // (tile - overlap) + (1 if (w - overlap) % (tile - overlap) > 0 else 0))
        tiles_y = max(1, (h - overlap) // (tile - overlap) + (1 if (h - overlap) % (tile - overlap) > 0 else 0))

        total_tiles = tiles_x * tiles_y
        tile_count = 0

        for y in range(tiles_y):
            for x in range(tiles_x):
                # Calculate source tile bounds
                x0 = x * (tile - overlap) if x > 0 else 0
                y0 = y * (tile - overlap) if y > 0 else 0
                x1 = min(x0 + tile, w)
                y1 = min(y0 + tile, h)

                # Extract tile
                tile_img = self._image_rgb[y0:y1, x0:x1]

                # Upscale tile
                tile_tensor = torch.from_numpy(tile_img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
                with torch.no_grad():
                    tile_out = self._model.model(tile_tensor)
                tile_result = (tile_out.squeeze().permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

                # Calculate output bounds
                out_x0, out_y0 = x0 * scale, y0 * scale
                out_x1, out_y1 = x1 * scale, y1 * scale

                # Blend overlap regions with simple linear blend
                blend_margin = overlap * scale // 2

                # For edge tiles, just copy
                if x == 0 and y == 0:
                    result[out_y0:out_y1, out_x0:out_x1] = tile_result
                else:
                    # Simple copy for now (Real-ESRGAN produces consistent results)
                    result[out_y0:out_y1, out_x0:out_x1] = tile_result

                tile_count += 1
                self.progress.emit(f"Processing tile {tile_count}/{total_tiles}...")

                # Clear GPU memory
                del tile_tensor, tile_out
                if device.type == 'mps':
                    torch.mps.empty_cache()

        return result


class SRService(QObject):
    model_loaded = Signal()
    model_load_failed = Signal(str)
    load_progress = Signal(str)
    upscale_complete = Signal(np.ndarray)
    upscale_failed = Signal(str)
    upscale_progress = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._model = None
        self._scale = 4
        self._is_loading = False
        self._is_loaded = False
        self._is_upscaling = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def is_loading(self) -> bool:
        return self._is_loading

    @property
    def is_upscaling(self) -> bool:
        return self._is_upscaling

    @property
    def scale(self) -> int:
        return self._scale

    def load_model(self, scale: int = 4):
        """Load Real-ESRGAN model (scale=2 or 4)."""
        if self._is_loading or (self._is_loaded and self._scale == scale):
            return
        self._scale = scale
        self._is_loading = True
        self._worker = SRLoadWorker(scale)
        self._worker.progress.connect(self.load_progress)
        self._worker.finished.connect(self._on_model_loaded)
        self._worker.error.connect(self._on_model_error)
        self._worker.start()

    def _on_model_loaded(self, result):
        self._model, self._scale = result
        self._is_loaded = True
        self._is_loading = False
        self.model_loaded.emit()

    def _on_model_error(self, error: str):
        self._is_loading = False
        self.model_load_failed.emit(error)

    def upscale(self, image_rgb: np.ndarray, params: dict):
        """Run super resolution on image.

        params:
            - scale: 2 or 4 (default: model's loaded scale)
            - tile_size: tile size for large images (default: 512)
        """
        if self._model is None:
            self.upscale_failed.emit("SR model not loaded")
            return
        if self._is_upscaling:
            self.upscale_failed.emit("SR already running")
            return

        self._is_upscaling = True
        scale = params.get("scale", self._scale)
        tile_size = params.get("tile_size", 512)

        # If scale differs from loaded model, reload
        if scale != self._scale:
            self._is_upscaling = False
            self.load_model(scale)
            # Will need to call upscale again after load completes
            return

        self._upscale_worker = SRUpscaleWorker(self._model, scale, image_rgb, tile_size)
        self._upscale_worker.progress.connect(self.upscale_progress)
        self._upscale_worker.finished.connect(self._on_upscale_finished)
        self._upscale_worker.error.connect(self._on_upscale_error)
        self._upscale_worker.start()

    def _on_upscale_finished(self, result):
        self._is_upscaling = False
        output, w, h, elapsed = result
        self.upscale_complete.emit(output)

    def _on_upscale_error(self, error: str):
        self._is_upscaling = False
        self.upscale_failed.emit(error)
