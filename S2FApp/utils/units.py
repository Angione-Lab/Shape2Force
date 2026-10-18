"""
Physical units for S2F predictions.

The network output is a normalised map y_hat in [0, 1]. The ground-truth traction-stress maps used for
training were exported by the TFM software on a fixed colour scale, so

    traction stress T [Pa] = y_hat * STRESS_MAX_PA[model_type]   (500 Pa single cells, 50 Pa spheroids).

Areas and forces additionally need the pixel size of the bright-field image (µm per pixel, entered by the
user). The prediction is made on a MODEL_INPUT_SIZE x MODEL_INPUT_SIZE grid, so one heatmap pixel covers
(image width * pixel size / MODEL_INPUT_SIZE) µm in x, and likewise in y.

    area  [µm²] = number of pixels * heatmap pixel area
    force [nN]  = sum(T [Pa] * heatmap pixel area [µm²]) * 1e-3      (1 Pa · µm² = 1e-12 N = 1e-3 nN)
"""
import numpy as np

from config.constants import MODEL_INPUT_SIZE, STRESS_MAX_PA


def stress_max_pa(model_type):
    return STRESS_MAX_PA["spheroid" if model_type == "spheroid" else "single_cell"]


def make_scale(model_type, image_um_per_px=None, image_shape=None, heatmap_shape=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)):
    """
    Physical scale of one prediction.

    image_um_per_px: pixel size of the uploaded bright-field image (µm/px); None or <= 0 if unknown.
    image_shape: (height, width) of the uploaded image, before resizing to the model input.
    Returns a dict used by the display, report and measurement code.
    """
    smax = stress_max_pa(model_type)
    s = {"stress_max_pa": smax, "image_um_per_px": None, "image_shape": None,
         "hm_um_per_px_x": None, "hm_um_per_px_y": None, "hm_px_area_um2": None}
    if image_shape is not None:
        s["image_shape"] = (int(image_shape[0]), int(image_shape[1]))
    if image_um_per_px and image_um_per_px > 0 and image_shape is not None:
        h, w = image_shape[:2]
        hh, hw = heatmap_shape[:2]
        sx = image_um_per_px * w / hw
        sy = image_um_per_px * h / hh
        s.update(image_um_per_px=float(image_um_per_px), hm_um_per_px_x=sx, hm_um_per_px_y=sy, hm_px_area_um2=sx * sy)
    return s


def has_physical_size(scale):
    return scale is not None and scale.get("hm_px_area_um2") is not None


def to_pa(heatmap, scale):
    """Normalised map -> traction stress in Pa."""
    return np.clip(np.asarray(heatmap, dtype=np.float64), 0, 1) * scale["stress_max_pa"]


def region_metrics(heatmap, mask, scale):
    """
    Metrics over a region (mask = 1) or the whole map (mask = None).
    Keys: n_px, area_um2, mean_pa, max_pa, total_force_nN (area and force are None without a pixel size).
    """
    t = to_pa(heatmap, scale)
    m = np.ones(t.shape, dtype=bool) if mask is None else (np.asarray(mask) > 0)
    n = int(m.sum())
    if n == 0:
        return {"n_px": 0, "area_um2": None, "mean_pa": None, "max_pa": None, "total_force_nN": None}
    vals = t[m]
    out = {"n_px": n, "mean_pa": float(vals.mean()), "max_pa": float(vals.max()), "area_um2": None, "total_force_nN": None}
    if has_physical_size(scale):
        a = scale["hm_px_area_um2"]
        out["area_um2"] = n * a
        out["total_force_nN"] = float(vals.sum()) * a * 1e-3
    return out


def fmt(value, unit="", nd=1):
    """Format a metric value; '–' when not available."""
    if value is None:
        return "–"
    return f"{value:.{nd}f}{(' ' + unit) if unit else ''}"


def metric_rows(metrics):
    """(label, formatted value) pairs, in the order used everywhere in the app."""
    return [
        ("Mean traction stress", fmt(metrics.get("mean_pa"), "Pa")),
        ("Max traction stress", fmt(metrics.get("max_pa"), "Pa")),
        ("Area", fmt(metrics.get("area_um2"), "µm²", 0)),
        ("Total force", fmt(metrics.get("total_force_nN"), "nN", 2)),
    ]


def nice_scale_bar_um(field_width_um, fraction=0.2):
    """Scale-bar length (µm) from {1, 2, 5} x 10^k closest to `fraction` of the field width."""
    target = field_width_um * fraction
    if target <= 0:
        return None
    k = np.floor(np.log10(target))
    candidates = [m * 10 ** k for m in (1, 2, 5, 10)]
    return float(min(candidates, key=lambda c: abs(c - target)))


def draw_scale_bar(rgb, um_per_px, color=(255, 255, 255), margin_frac=0.04):
    """Draw a µm scale bar (bottom right) on an RGB uint8 image in place. Returns (rgb, length_um)."""
    import cv2

    if not um_per_px or um_per_px <= 0:
        return rgb, None
    h, w = rgb.shape[:2]
    length_um = nice_scale_bar_um(w * um_per_px)
    if length_um is None:
        return rgb, None
    length_px = int(round(length_um / um_per_px))
    thick = max(2, h // 120)
    m = int(margin_frac * w)
    x1, y = w - m, h - m
    x0 = x1 - length_px
    cv2.rectangle(rgb, (x0 - 1, y - thick - 1), (x1 + 1, y + 1), (0, 0, 0), -1)
    cv2.rectangle(rgb, (x0, y - thick), (x1, y), color, -1)
    label = f"{length_um:g} um"   # OpenCV fonts have no µ glyph
    font_scale = max(0.4, w / 1100)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, thick // 2))
    tx, ty = x0 + (length_px - tw) // 2, y - thick - 6
    cv2.putText(rgb, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), max(2, thick), cv2.LINE_AA)
    cv2.putText(rgb, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, max(1, thick // 2), cv2.LINE_AA)
    return rgb, length_um
