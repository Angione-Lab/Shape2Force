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


def is_display_range_remapped(display_mode, clip_min, clip_max):
    """
    True when the UI uses Range mode with a clip window other than the full [0, 1]
    (colorbar tick labels map normalized 0–1 to that interval).
    """
    if display_mode != "Range":
        return False
    lo, hi = float(clip_min), float(clip_max)
    return hi > lo and not (lo == 0.0 and hi == 1.0)


def apply_display_scale(heatmap, mode, clip_min=0, clip_max=1, clamp_only=False):
    """
    Apply display scaling. Display only—does not change underlying values.

    Supports modes used by the app: ``Default`` (clip to [0, 1]) and ``Range``
    (window to [clip_min, clip_max]). Unknown ``mode`` is treated like ``Default``.

    Parameters
    ----------
    clamp_only : bool
        For ``mode == "Range"`` when ``clip_max > clip_min``:

        - **False** (default in the app for Range): pixels outside ``[clip_min, clip_max]`` are set
          to 0; values inside are linearly mapped to ``[0, 1]`` for the colormap.
        - **True**: values are clamped to ``[clip_min, clip_max]`` without rescaling to ``[0, 1]``.
    """
    if mode != "Range":
        return np.clip(heatmap, 0, 1).astype(np.float32)
    # Range
    vmin, vmax = float(clip_min), float(clip_max)
    if vmax > vmin:
        h = heatmap.astype(np.float32)
        if clamp_only:
            return np.clip(h, vmin, vmax).astype(np.float32)
        mask = (h >= vmin) & (h <= vmax)
        out = np.zeros_like(h)
        out[mask] = (h[mask] - vmin) / (vmax - vmin)
        return np.clip(out, 0, 1).astype(np.float32)
    return np.clip(heatmap, 0, 1).astype(np.float32)
