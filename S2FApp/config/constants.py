"""
Centralized constants for S2F App.
"""
import cv2

# Model & paths
MODEL_INPUT_SIZE = 1024

APP_VERSION = "1.1.0"   # shown in reports; the git commit is added from S2F_APP_COMMIT if set

# Default substrate (used when config lookup fails or manual mode fallback).
# 6 kPa reference setting: the released single-cell model is most accurate with it for every image
# (sensitivity analysis in the paper's revision, Fig. S4), so it is the recommended input.
DEFAULT_SUBSTRATE = "Fibroblasts_Fibronectin_6KPa"
REFERENCE_SUBSTRATE = DEFAULT_SUBSTRATE

# Normalisation of the two substrate channels: the ranges used in training (S2F config/substrate_settings.json).
# Fixed here so that the model input does not depend on which presets the app lists.
SETTINGS_NORMALIZATION = {"pixelsize": {"min": 3.0769, "max": 9.8138}, "young": {"min": 1000.0, "max": 10000.0}}

# Physical scale of the output: the ground-truth maps were exported on a fixed colour scale,
# so a normalised value of 1 corresponds to this traction stress.
STRESS_MAX_PA = {"single_cell": 500.0, "spheroid": 50.0}

# Default pixel size of the bright-field image (µm/px) offered in the sidebar; None = ask the user.
# 0.325 µm/px is the pixel size of the 2048 x 2048 single-cell training images (TFM setting 3.0769 px/µm).
DEFAULT_IMAGE_UM_PER_PX = {"single_cell": 0.325, "spheroid": None}

# Spheroid checkpoints: default = the smaller supervised model (FN), which gives smoother maps and similar accuracy
# (validation, 28 spheroids: pixel r 0.88 vs 0.88, RME 0.44 vs 0.47, MS-SSIM 0.79 vs 0.74 for the S2F/GN model)
SPHEROID_DEFAULT_CKP = "ckp_spheroid_FN.pth"
ARCH_LABELS = {
    "s2f": "S2F generator (attention U-Net + PatchGAN training), 77.0 M parameters",
    "s2f_spheroid": "Supervised attention U-Net (no discriminator), 11.8 M parameters",
}

# Auto-boundary parameters (utils/segmentation.estimate_cell_mask defaults), recorded in reports
AUTO_BOUNDARY_DESCRIPTION = "high-stress region from the predicted map: Otsu threshold x 0.85 after Gaussian smoothing (sigma 2 px), dilated by 4 px"

# UI
CANVAS_SIZE = 320
SAMPLE_THUMBNAIL_LIMIT = 8
BATCH_MAX_IMAGES = 5
# Max images per model forward pass (avoids OOM on Hugging Face free tier)
BATCH_INFERENCE_SIZE = 2
COLORMAP_N_SAMPLES = 64

# Model type labels
MODEL_TYPE_LABELS = {"single_cell": "Single cell", "spheroid": "Spheroid LS174T"}

# Drawing tools
DRAW_TOOLS = ["polygon", "rect", "circle"]
TOOL_LABELS = {"polygon": "Polygon", "rect": "Rectangle", "circle": "Circle"}

# File extensions
SAMPLE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

# Colormaps (OpenCV)
COLORMAPS = {
    "Jet": cv2.COLORMAP_JET,
    "Viridis": cv2.COLORMAP_VIRIDIS,
    "Plasma": cv2.COLORMAP_PLASMA,
    "Inferno": cv2.COLORMAP_INFERNO,
    "Magma": cv2.COLORMAP_MAGMA,
}
