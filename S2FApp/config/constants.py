"""
Centralized constants for S2F App.
"""
import cv2

# Model & paths
MODEL_INPUT_SIZE = 1024

# UI
CANVAS_SIZE = 320
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
