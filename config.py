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

# Super Resolution (Real-ESRGAN - lightweight CNN upscaler)
SR_MODEL_DIR = Path(__file__).parent / "sr_models"
SR_DEFAULT_SCALE = 4
SR_SCALE_CHOICES = [2, 4]
SR_DEFAULT_TILE_SIZE = 512