from pathlib import Path

# SAM2 large model (more accurate, slower than tiny)
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CHECKPOINT_NAME = "sam2.1_hiera_large.pt"

SAM2_DIR = Path(__file__).parent / "sam2"
CHECKPOINT_PATH = SAM2_DIR / "checkpoints" / SAM2_CHECKPOINT_NAME

# Alternative: tiny model (faster, less accurate)
# SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
# SAM2_CHECKPOINT_NAME = "sam2.1_hiera_tiny.pt"

POSITIVE_POINT_COLOR = (0, 220, 0)
NEGATIVE_POINT_COLOR = (220, 0, 0)
BBOX_COLOR = (0, 220, 0)
MASK_OVERLAY_COLOR = (255, 180, 0)
MASK_OVERLAY_ALPHA = 100
MASK_CONTOUR_COLOR = (255, 255, 0)
POINT_RADIUS = 7
MIN_BBOX_SIZE = 5

DEFAULT_MOSAIC_BLOCK_SIZE = 10
SAVE_DIR = Path(__file__).parent / "saved"

# Super Resolution (ControlNet Tile SR + SDXL)
SR_CONTROLNET_MODEL = "brad-twinkl/controlnet-union-sdxl-1.0-promax"
SR_SDXL_MODEL = "SG161222/RealVisXL_V5.0"
SR_VAE_MODEL = "madebyollin/sdxl-vae-fp16-fix"

SR_RESOLUTION_CHOICES = [1024, 2048, 4096, 8192]
SR_DEFAULT_RESOLUTION = 4096
SR_DEFAULT_STRENGTH = 0.65
SR_DEFAULT_GUIDANCE_SCALE = 4.0
SR_DEFAULT_NUM_STEPS = 35
SR_DEFAULT_MAX_TILE_SIZE = 1024
SR_DEFAULT_PROMPT = "high-quality, noise-free edges, high quality, 4k, hd, 8k"
SR_DEFAULT_NEGATIVE_PROMPT = "blurry, pixelated, noisy, low resolution, artifacts, poor details"