"""Heatmap visualization utilities (colorbar, overlays, Plotly)."""
import base64

import cv2
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from config.constants import COLORMAPS


def _colormap_gradient_base64(colormap_name, width=512):
    """Generate a horizontal gradient bar as base64 PNG for the given colormap."""
    cv2_cmap = COLORMAPS.get(colormap_name, cv2.COLORMAP_JET)
    gradient = np.linspace(0, 255, width, dtype=np.uint8).reshape(1, -1)
    rgb = cv2.cvtColor(cv2.applyColorMap(gradient, cv2_cmap), cv2.COLOR_BGR2RGB)
    bar = np.repeat(rgb, 6, axis=0)
    _, buf = cv2.imencode(".png", cv2.cvtColor(bar, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf.tobytes()).decode("utf-8")

# Distinct colors for each region (RGB - heatmap_rgb is RGB)
_REGION_COLORS = [
    (0, 188, 212),   # cyan (matches drawing tool)
    (0, 230, 118),   # green
    (255, 235, 59),  # yellow
    (171, 71, 188),  # purple
    (0, 150, 255),   # blue
    (255, 167, 38),  # amber
    (124, 179, 66),  # light green
    (233, 30, 99),   # pink
]


def _draw_region_overlay(annotated, mask, color, fill_alpha=0.3, stroke_width=2):
    """Draw single region overlay on annotated heatmap (fill + alpha blend + contour). Modifies annotated in place."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = annotated.copy()
    cv2.fillPoly(overlay, contours, color)
    mask_3d = np.stack([mask] * 3, axis=-1).astype(bool)
    annotated[mask_3d] = (
        (1 - fill_alpha) * annotated[mask_3d].astype(np.float32)
        + fill_alpha * overlay[mask_3d].astype(np.float32)
    ).astype(np.uint8)
    cv2.drawContours(annotated, contours, -1, color, stroke_width)


def render_horizontal_colorbar(colormap_name, clip_min=0, clip_max=1, is_rescale=False):
    """Render a compact horizontal colorbar for batch mode, anchored above the table."""
    ticks = [0, 0.25, 0.5, 0.75, 1]
    if is_rescale:
        rng = clip_max - clip_min
        labels = [f"{clip_min + t * rng:.2f}" for t in ticks]
    else:
        labels = [f"{t:.2f}" for t in ticks]

    data_url = _colormap_gradient_base64(colormap_name)
    labels_html = "".join(f'<span class="cb-tick">{l}</span>' for l in labels)
    html = f"""
    <div class="colorbar-table-header">
        <div class="colorbar-ticks">{labels_html}</div>
        <div class="colorbar-bar" style="background-image: url(data:image/png;base64,{data_url});"></div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def make_annotated_heatmap(heatmap_rgb, mask, fill_alpha=0.3, stroke_color=(0, 188, 212), stroke_width=2):
    """Composite heatmap with drawn region overlay."""
    annotated = heatmap_rgb.copy()
    _draw_region_overlay(annotated, mask, stroke_color, fill_alpha, stroke_width)
    return annotated


def make_annotated_heatmap_multi_regions(heatmap_rgb, masks, labels, cell_mask=None, fill_alpha=0.3):
    """Draw each region separately with distinct color and label (R1, R2, ...). No merging."""
    annotated = heatmap_rgb.copy()
    if cell_mask is not None and np.any(cell_mask > 0):
        contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, contours, -1, (255, 0, 0), 3)
    for i, mask in enumerate(masks):
        color = _REGION_COLORS[i % len(_REGION_COLORS)]
        _draw_region_overlay(annotated, mask, color, fill_alpha, stroke_width=2)
        # Label at centroid
        M = cv2.moments(mask)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            label = labels[i] if i < len(labels) else f"R{i + 1}"
            cv2.putText(
                annotated, label, (cx - 12, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
            )
            cv2.putText(
                annotated, label, (cx - 12, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA
            )
    return annotated


def add_cell_contour_to_fig(fig_pl, cell_mask, row=1, col=2):
    """Add red contour overlay to Plotly heatmap subplot. Draws all contours (handles multiple disconnected regions)."""
    if cell_mask is None or not np.any(cell_mask > 0):
        return
    contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return
    for cnt in contours:
        pts = cnt.squeeze()
        if pts.ndim == 1:
            pts = pts.reshape(1, 2)
        x, y = pts[:, 0].tolist(), pts[:, 1].tolist()
        if x[0] != x[-1] or y[0] != y[-1]:
            x.append(x[0])
            y.append(y[0])
        fig_pl.add_trace(
            go.Scatter(x=x, y=y, mode="lines", line=dict(color="red", width=3), showlegend=False),
            row=row, col=col
        )
