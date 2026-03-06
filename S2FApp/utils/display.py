"""Display utilities for heatmaps and colormaps."""
import numpy as np
import cv2

from config.constants import COLORMAPS, COLORMAP_N_SAMPLES


def cv_colormap_to_plotly_colorscale(colormap_name, n_samples=None):
    """Build a Plotly colorscale from OpenCV colormap so UI matches download/PDF exactly."""
    n = n_samples or COLORMAP_N_SAMPLES
    cv2_cmap = COLORMAPS.get(colormap_name, cv2.COLORMAP_JET)
    gradient = np.linspace(0, 255, n, dtype=np.uint8).reshape(1, -1)
    rgb = cv2.applyColorMap(gradient, cv2_cmap)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    scale = []
    for i in range(n):
        r, g, b = rgb[0, i]
        scale.append([i / (n - 1), f"rgb({r},{g},{b})"])
    return scale


def apply_display_scale(heatmap, mode):
    """
    Apply display scaling (Fiji-style). Display only—does not change underlying values.
    - Auto: map data min..max to 0..1 (full color range)
    - Fixed: use 0-1 range as-is
    """
    if mode == "Fixed":
        return np.clip(heatmap, 0, 1).astype(np.float32)
    hmin, hmax = float(np.min(heatmap)), float(np.max(heatmap))
    if hmax > hmin:
        out = (heatmap.astype(np.float32) - hmin) / (hmax - hmin)
        return np.clip(out, 0, 1).astype(np.float32)
    return np.clip(heatmap, 0, 1).astype(np.float32)
