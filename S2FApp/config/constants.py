"""
Centralized constants for S2F App.
"""
import cv2

# Model & paths
MODEL_INPUT_SIZE = 1024

# Default substrate (used when config lookup fails or manual mode fallback)
DEFAULT_SUBSTRATE = "Fibroblasts_Fibronectin_6KPa"

# UI
CANVAS_SIZE = 320
SAMPLE_THUMBNAIL_LIMIT = 8
BATCH_MAX_IMAGES = 5
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
