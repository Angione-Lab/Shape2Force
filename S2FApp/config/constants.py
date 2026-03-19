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

# UI themes: primary, primary-dark, primary-darker, rgb (for rgba)
THEMES = {
    "Teal": ("#0d9488", "#0f766e", "#115e59", "13, 148, 136"),
    "Blue": ("#2563eb", "#1d4ed8", "#1e40af", "37, 99, 235"),
    "Indigo": ("#6366f1", "#4f46e5", "#4338ca", "99, 102, 241"),
    "Purple": ("#7c3aed", "#6d28d9", "#5b21b6", "124, 58, 237"),
    "Amber": ("#f59e0b", "#d97706", "#b45309", "245, 158, 11"),
    "Emerald": ("#10b981", "#059669", "#047857", "16, 185, 129"),
}

# Colormaps (OpenCV)
COLORMAPS = {
    "Jet": cv2.COLORMAP_JET,
    "Viridis": cv2.COLORMAP_VIRIDIS,
    "Plasma": cv2.COLORMAP_PLASMA,
    "Inferno": cv2.COLORMAP_INFERNO,
    "Magma": cv2.COLORMAP_MAGMA,
}
