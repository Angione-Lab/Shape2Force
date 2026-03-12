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


def apply_display_scale(heatmap, mode, min_percentile=0, max_percentile=100, clip_min=0, clip_max=1):
    """
    Apply display scaling (Fiji/ImageJ style). Display only—does not change underlying values.
    - Default: full 0–1 range as-is.
    - Percentile: map min..max percentiles to 0..1.
    - Range: show only values in [clip_min, clip_max]; others hidden (black).
    """
    if mode == "Default" or mode == "Auto" or mode == "Full":
        return np.clip(heatmap, 0, 1).astype(np.float32)
    if mode == "Percentile":
        pmin = float(np.percentile(heatmap, min_percentile))
        pmax = float(np.percentile(heatmap, max_percentile))
        if pmax > pmin:
            out = (heatmap.astype(np.float32) - pmin) / (pmax - pmin)
            return np.clip(out, 0, 1).astype(np.float32)
        return np.clip(heatmap, 0, 1).astype(np.float32)
    if mode == "Range" or mode == "Filter" or mode == "Threshold":
        vmin, vmax = float(clip_min), float(clip_max)
        if vmax > vmin:
            h = heatmap.astype(np.float32)
            mask = (h >= vmin) & (h <= vmax)
            out = np.zeros_like(h)
            out[mask] = (h[mask] - vmin) / (vmax - vmin)
            return np.clip(out, 0, 1).astype(np.float32)
        return np.clip(heatmap, 0, 1).astype(np.float32)
    return np.clip(heatmap, 0, 1).astype(np.float32)
