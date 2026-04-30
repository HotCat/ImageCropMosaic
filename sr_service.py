import time
from typing import Optional

import numpy as np
import torch
from PIL import Image
from PySide6.QtCore import QObject, Signal, QThread

from config import (
    SR_CONTROLNET_MODEL, SR_SDXL_MODEL, SR_VAE_MODEL,
)


class SRLoadWorker(QThread):
    finished = Signal(object)
    progress = Signal(str)
    error = Signal(str)

    def run(self):
        try:
            self.progress.emit("Determining device...")
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

            self.progress.emit("Loading VAE...")
            from diffusers import AutoencoderKL
            vae = AutoencoderKL.from_pretrained(
                SR_VAE_MODEL, torch_dtype=torch.float32
            ).to(device)

            self.progress.emit("Loading ControlNet Union model...")
            from diffusers import ControlNetUnionModel
            controlnet = ControlNetUnionModel.from_pretrained(
                SR_CONTROLNET_MODEL, torch_dtype=torch.float32
            ).to(device)

            self.progress.emit("Loading SDXL pipeline (may download ~3GB first run)...")
            from sr_pipeline import StableDiffusionXLControlNetTileSRPipeline
            from diffusers import UniPCMultistepScheduler

            pipe = StableDiffusionXLControlNetTileSRPipeline.from_pretrained(
                SR_SDXL_MODEL,
                controlnet=controlnet,
                vae=vae,
                torch_dtype=torch.float32,
                use_safetensors=True,
            ).to(device)

            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

            self.progress.emit("Enabling memory optimizations...")
            try:
                pipe.enable_vae_tiling()
                pipe.enable_vae_slicing()
            except Exception:
                pass

            self.progress.emit("SR model loaded successfully")
            self.finished.emit(pipe)

        except Exception as e:
            self.error.emit(str(e))


class SRUpscaleWorker(QThread):
    finished = Signal(object)
    progress = Signal(str)
    error = Signal(str)

    def __init__(self, pipe, image_rgb: np.ndarray, params: dict, parent=None):
        super().__init__(parent)
        self._pipe = pipe
        self._image_rgb = image_rgb
        self._params = params

    def run(self):
        try:
            t0 = time.perf_counter()
            image = Image.fromarray(self._image_rgb)
            original_w, original_h = image.size
            params = self._params

            resolution = params.get("resolution", 4096)
            current_max = max(image.size)
            scale_factor = max(2, resolution / current_max)
            new_w = int(original_w * scale_factor)
            new_h = int(original_h * scale_factor)

            self.progress.emit(
                f"SR: Pre-upscaling {original_w}x{original_h} -> {new_w}x{new_h}..."
            )
            upscaled_input = image.resize((new_w, new_h), Image.LANCZOS)

            normal_overlap, border_overlap = self._pipe.calculate_overlap(new_w, new_h)

            self.progress.emit(
                f"SR: Running diffusion ({params.get('num_inference_steps', 35)} steps, "
                f"strength {params.get('strength', 0.65)})..."
            )

            result = self._pipe(
                image=upscaled_input,
                control_image=image,
                control_mode=[6],
                prompt=params.get("prompt", "high-quality, noise-free edges, high quality, 4k, hd, 8k"),
                negative_prompt=params.get("negative_prompt", "blurry, pixelated, noisy, low resolution, artifacts"),
                height=new_h,
                width=new_w,
                original_size=(original_w, original_h),
                target_size=(new_w, new_h),
                strength=float(params.get("strength", 0.65)),
                guidance_scale=float(params.get("guidance_scale", 4.0)),
                num_inference_steps=int(params.get("num_inference_steps", 35)),
                controlnet_conditioning_scale=1.0,
                normal_tile_overlap=normal_overlap,
                border_tile_overlap=border_overlap,
                max_tile_size=int(params.get("max_tile_size", 1024)),
                tile_weighting_method="Cosine",
                output_type="np",
            )

            output = result.images[0] if hasattr(result, 'images') else result[0]
            elapsed = time.perf_counter() - t0
            self.finished.emit((output, new_w, new_h, elapsed))

        except Exception as e:
            self.error.emit(str(e))


class SRService(QObject):
    model_loaded = Signal()
    model_load_failed = Signal(str)
    load_progress = Signal(str)
    upscale_complete = Signal(np.ndarray)
    upscale_failed = Signal(str)
    upscale_progress = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pipe = None
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

    def load_model(self):
        if self._is_loading or self._is_loaded:
            return
        self._is_loading = True
        self._worker = SRLoadWorker()
        self._worker.progress.connect(self.load_progress)
        self._worker.finished.connect(self._on_model_loaded)
        self._worker.error.connect(self._on_model_error)
        self._worker.start()

    def _on_model_loaded(self, pipe):
        self._pipe = pipe
        self._is_loaded = True
        self._is_loading = False
        self.model_loaded.emit()

    def _on_model_error(self, error: str):
        self._is_loading = False
        self.model_load_failed.emit(error)

    def upscale(self, image_rgb: np.ndarray, params: dict):
        if self._pipe is None:
            self.upscale_failed.emit("SR model not loaded")
            return
        if self._is_upscaling:
            self.upscale_failed.emit("SR already running")
            return

        self._is_upscaling = True
        self._upscale_worker = SRUpscaleWorker(self._pipe, image_rgb, params)
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