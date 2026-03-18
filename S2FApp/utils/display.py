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


def build_range_colorscale(colormap_name, clip_min, clip_max, n_range_samples=32):
    """
    Build a Plotly colorscale for Range mode: normal gradient in [clip_min, clip_max],
    the "zero" color everywhere else (0 → clip_min and clip_max → 1).
    """
    cv2_cmap = COLORMAPS.get(colormap_name, cv2.COLORMAP_JET)

    zero_px = np.array([[0]], dtype=np.uint8)
    zero_rgb = cv2.applyColorMap(zero_px, cv2_cmap)
    zero_rgb = cv2.cvtColor(zero_rgb, cv2.COLOR_BGR2RGB)
    zr, zg, zb = zero_rgb[0, 0]
    zero_color = f"rgb({zr},{zg},{zb})"

    eps = 0.0005
    scale = []

    scale.append([0.0, zero_color])
    if clip_min > eps:
        scale.append([clip_min - eps, zero_color])

    positions = np.linspace(clip_min, clip_max, n_range_samples)
    pixel_vals = np.clip((positions * 255).astype(np.uint8), 0, 255).reshape(1, -1)
    rgb = cv2.applyColorMap(pixel_vals, cv2_cmap)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    for i, pos in enumerate(positions):
        r, g, b = rgb[0, i]
        scale.append([float(pos), f"rgb({r},{g},{b})"])

    if clip_max < 1.0 - eps:
        scale.append([clip_max + eps, zero_color])
    scale.append([1.0, zero_color])

    return scale


def apply_display_scale(heatmap, mode, min_percentile=0, max_percentile=100,
                        clip_min=0, clip_max=1, clip_bounds=False):
    """
    Apply display scaling. Display only—does not change underlying values.
    - Default: full 0–1 range as-is.
    - Range: keep original values inside [clip_min, clip_max].
      clip_bounds=False → zero out outside.  clip_bounds=True → clamp to bounds.
    - Rescale: map [clip_min, clip_max] → [0, 1].
      clip_bounds=False → zero out outside.  clip_bounds=True → clamp to bounds first.
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
        # Range: filter (discard outside) + rescale [clip_min, clip_max] → [0, 1] so max shows as red
        vmin, vmax = float(clip_min), float(clip_max)
        if vmax > vmin:
            h = heatmap.astype(np.float32)
            if clip_bounds:
                return np.clip(h, vmin, vmax).astype(np.float32)
            mask = (h >= vmin) & (h <= vmax)
            out = np.zeros_like(h)
            out[mask] = (h[mask] - vmin) / (vmax - vmin)
            return np.clip(out, 0, 1).astype(np.float32)
        return np.clip(heatmap, 0, 1).astype(np.float32)
    if mode == "Rescale":
        vmin, vmax = float(clip_min), float(clip_max)
        if vmax > vmin:
            h = heatmap.astype(np.float32)
            if clip_bounds:
                clamped = np.clip(h, vmin, vmax)
                return ((clamped - vmin) / (vmax - vmin)).astype(np.float32)
            mask = (h >= vmin) & (h <= vmax)
            out = np.zeros_like(h)
            out[mask] = (h[mask] - vmin) / (vmax - vmin)
            return np.clip(out, 0, 1).astype(np.float32)
        return np.clip(heatmap, 0, 1).astype(np.float32)
    return np.clip(heatmap, 0, 1).astype(np.float32)
