"""Measure tool: drawable canvas, region metrics, and downloads."""
import csv
import html
import io
import os

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from config.constants import CANVAS_SIZE, DRAW_TOOLS, TOOL_LABELS
from utils.report import heatmap_to_rgb_with_contour, create_measure_pdf_report
from ui.heatmaps import make_annotated_heatmap_multi_regions

try:
    from streamlit_drawable_canvas import st_canvas
    HAS_DRAWABLE_CANVAS = True
except (ImportError, AttributeError):
    HAS_DRAWABLE_CANVAS = False


def _obj_to_pts(obj, scale_x, scale_y, heatmap_w, heatmap_h):
    """Convert a single canvas object to polygon points in heatmap coords. Returns None if invalid."""
    obj_type = obj.get("type", "")
    pts = []
    if obj_type == "rect":
        left = obj.get("left", 0)
        top = obj.get("top", 0)
        w = obj.get("width", 0)
        h = obj.get("height", 0)
        pts = np.array([
            [left, top], [left + w, top], [left + w, top + h], [left, top + h]
        ], dtype=np.float32)
    elif obj_type == "circle" or obj_type == "ellipse":
        left = obj.get("left", 0)
        top = obj.get("top", 0)
        width = obj.get("width", 0)
        height = obj.get("height", 0)
        radius = obj.get("radius", 0)
        angle_deg = obj.get("angle", 0)
        if radius > 0:
            rx = ry = radius
            angle_rad = np.deg2rad(angle_deg)
            cx = left + radius * np.cos(angle_rad)
            cy = top + radius * np.sin(angle_rad)
        else:
            rx = width / 2 if width > 0 else 0
            ry = height / 2 if height > 0 else 0
            if rx <= 0 or ry <= 0:
                return None
            cx = left + rx
            cy = top + ry
        if rx <= 0 or ry <= 0:
            return None
        n = 32
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts = np.column_stack([cx + rx * np.cos(angles), cy + ry * np.sin(angles)]).astype(np.float32)
    elif obj_type == "path":
        path = obj.get("path", [])
        for cmd in path:
            if isinstance(cmd, (list, tuple)) and len(cmd) >= 3:
                if cmd[0] in ("M", "L"):
                    pts.append([float(cmd[1]), float(cmd[2])])
                elif cmd[0] == "Q" and len(cmd) >= 5:
                    pts.append([float(cmd[3]), float(cmd[4])])
                elif cmd[0] == "C" and len(cmd) >= 7:
                    pts.append([float(cmd[5]), float(cmd[6])])
        if len(pts) < 3:
            return None
        pts = np.array(pts, dtype=np.float32)
    else:
        return None
    pts[:, 0] *= scale_x
    pts[:, 1] *= scale_y
    pts = np.clip(pts, 0, [heatmap_w - 1, heatmap_h - 1]).astype(np.int32)
    return pts


def parse_canvas_shapes_to_masks(json_data, canvas_h, canvas_w, heatmap_h, heatmap_w):
    """Parse drawn shapes and return a list of individual masks (one per shape)."""
    if not json_data or "objects" not in json_data or not json_data["objects"]:
        return []
    scale_x = heatmap_w / canvas_w
    scale_y = heatmap_h / canvas_h
    masks = []
    for obj in json_data["objects"]:
        pts = _obj_to_pts(obj, scale_x, scale_y, heatmap_w, heatmap_h)
        if pts is None:
            continue
        mask = np.zeros((heatmap_h, heatmap_w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)
        masks.append(mask)
    return masks


def build_original_vals(raw_heatmap, pixel_sum, force):
    """Build original_vals dict for measure tool (full map)."""
    return {
        "pixel_sum": pixel_sum,
        "force": force,
        "max": float(np.max(raw_heatmap)),
        "mean": float(np.mean(raw_heatmap)),
    }


def _compute_cell_metrics(raw_heatmap, cell_mask, pixel_sum, force):
    """Compute metrics over estimated cell area only."""
    area_px = int(np.sum(cell_mask))
    if area_px == 0:
        return None, None, None
    region_values = raw_heatmap * cell_mask
    cell_pixel_sum = float(np.sum(region_values))
    cell_force = cell_pixel_sum * (force / pixel_sum) if pixel_sum > 0 else cell_pixel_sum
    cell_mean = cell_pixel_sum / area_px
    return cell_pixel_sum, cell_force, cell_mean


def build_cell_vals(raw_heatmap, cell_mask, pixel_sum, force):
    """Build cell_vals dict for measure tool (estimated cell area). Returns None if invalid."""
    cell_pixel_sum, cell_force, cell_mean = _compute_cell_metrics(raw_heatmap, cell_mask, pixel_sum, force)
    if cell_pixel_sum is None:
        return None
    region_values = raw_heatmap * cell_mask
    region_nonzero = region_values[cell_mask > 0]
    cell_max = float(np.max(region_nonzero)) if len(region_nonzero) > 0 else 0
    return {
        "pixel_sum": cell_pixel_sum,
        "force": cell_force,
        "max": cell_max,
        "mean": cell_mean,
    }


def compute_region_metrics(raw_heatmap, mask, original_vals=None):
    """Compute region metrics from mask."""
    area_px = int(np.sum(mask))
    region_values = raw_heatmap * mask
    region_nonzero = region_values[mask > 0]
    force_sum = float(np.sum(region_values))
    density = force_sum / area_px if area_px > 0 else 0
    region_max = float(np.max(region_nonzero)) if len(region_nonzero) > 0 else 0
    region_mean = float(np.mean(region_nonzero)) if len(region_nonzero) > 0 else 0
    region_force_scaled = (
        force_sum * (original_vals["force"] / original_vals["pixel_sum"])
        if original_vals and original_vals.get("pixel_sum", 0) > 0
        else force_sum
    )
    return {
        "area_px": area_px,
        "force_sum": force_sum,
        "density": density,
        "max": region_max,
        "mean": region_mean,
        "force_scaled": region_force_scaled,
    }


def _draw_contour_on_image(img_rgb, mask, stroke_color=(255, 0, 0), stroke_width=3):
    """Draw contour from mask on RGB image. Resizes mask to match img if needed."""
    h, w = img_rgb.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(img_rgb, contours, -1, stroke_color, stroke_width)
    return img_rgb


def render_region_metrics_and_downloads(metrics_list, masks, heatmap_rgb, input_filename, key_suffix, has_original_vals,
                                        first_region_label=None, bf_img=None, cell_mask=None, colormap_name="Jet"):
    """
    Render per-shape metrics table and download buttons.
    first_region_label: custom label for first row (e.g. 'Auto boundary').
    masks: list of region masks (user-drawn only; used for labeled heatmap with R1, R2...).
    """
    base_name = os.path.splitext(input_filename or "image")[0]
    st.markdown("**Regions (each selection = one row)**")
    if has_original_vals:
        headers = ["Region", "Area", "F.sum", "Force", "Max", "Mean"]
        csv_rows = [["image", "region"] + headers[1:]]
    else:
        headers = ["Region", "Area (px²)", "Force sum", "Mean"]
        csv_rows = [["image", "region", "Area", "Force sum", "Mean"]]
    table_rows = [headers]
    for i, metrics in enumerate(metrics_list, 1):
        region_label = first_region_label if (i == 1 and first_region_label) else f"Region {i - (1 if first_region_label else 0)}"
        if has_original_vals:
            row = [region_label, str(metrics["area_px"]), f"{metrics['force_sum']:.3f}", f"{metrics['force_scaled']:.1f}",
                   f"{metrics['max']:.3f}", f"{metrics['mean']:.4f}"]
            csv_rows.append([base_name, region_label, metrics["area_px"], f"{metrics['force_sum']:.3f}",
                            f"{metrics['force_scaled']:.1f}", f"{metrics['max']:.3f}", f"{metrics['mean']:.4f}"])
        else:
            row = [region_label, str(metrics["area_px"]), f"{metrics['force_sum']:.4f}", f"{metrics['mean']:.6f}"]
            csv_rows.append([base_name, region_label, metrics["area_px"], f"{metrics['force_sum']:.4f}",
                            f"{metrics['mean']:.6f}"])
        table_rows.append(row)
    # Render as HTML table to avoid Streamlit's default row/column indices
    header = table_rows[0]
    body = table_rows[1:]
    th_cells = "".join(
        f'<th style="border: 1px solid #ddd; padding: 8px; text-align: left;">{html.escape(str(h))}</th>'
        for h in header
    )
    rows_html = [
        "<tr>"
        + "".join(
            f'<td style="border: 1px solid #ddd; padding: 8px;">{html.escape(str(c))}</td>'
            for c in row
        )
        + "</tr>"
        for row in body
    ]
    table_html = (
        f'<table style="border-collapse: collapse; width: 100%;">'
        f"<thead><tr>{th_cells}</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody></table>"
    )
    st.markdown(table_html, unsafe_allow_html=True)
    buf_csv = io.StringIO()
    csv.writer(buf_csv).writerows(csv_rows)
    # Annotated heatmap: each region separate with R1, R2 labels (no merging)
    region_labels = [f"R{i + 1}" for i in range(len(masks))]
    heatmap_labeled = make_annotated_heatmap_multi_regions(heatmap_rgb.copy(), masks, region_labels, cell_mask=None)
    buf_img = io.BytesIO()
    Image.fromarray(heatmap_labeled).save(buf_img, format="PNG")
    buf_img.seek(0)
    # PDF report (requires bf_img)
    pdf_bytes = None
    if bf_img is not None:
        pdf_bytes = create_measure_pdf_report(bf_img, heatmap_labeled, table_rows, base_name)
    n_cols = 3 if pdf_bytes is not None else 2
    dl_cols = st.columns(n_cols)
    with dl_cols[0]:
        st.download_button("Download all regions", data=buf_csv.getvalue(),
            file_name=f"{base_name}_all_regions.csv", mime="text/csv",
            key=f"download_all_regions_{key_suffix}", icon=":material/download:")
    with dl_cols[1]:
        st.download_button("Download heatmap", data=buf_img.getvalue(),
            file_name=f"{base_name}_annotated_heatmap.png", mime="image/png",
            key=f"download_annotated_{key_suffix}", icon=":material/image:")
    if pdf_bytes is not None:
        with dl_cols[2]:
            st.download_button("Download report", data=pdf_bytes,
                file_name=f"{base_name}_measure_report.pdf", mime="application/pdf",
                key=f"download_measure_pdf_{key_suffix}", icon=":material/picture_as_pdf:")


def render_region_canvas(display_heatmap, raw_heatmap=None, bf_img=None, original_vals=None, cell_vals=None,
                         cell_mask=None, key_suffix="", input_filename=None, colormap_name="Jet"):
    """Render drawable canvas and region metrics. When cell_vals: show cell area (replaces Full map). Else: show Full map."""
    if not HAS_DRAWABLE_CANVAS:
        st.caption("Install `streamlit-drawable-canvas-fix` for region measurement: `pip install streamlit-drawable-canvas-fix`")
        return
    raw_heatmap = raw_heatmap if raw_heatmap is not None else display_heatmap
    h, w = display_heatmap.shape
    heatmap_rgb = heatmap_to_rgb_with_contour(display_heatmap, colormap_name, cell_mask)
    pil_bg = Image.fromarray(heatmap_rgb).resize((CANVAS_SIZE, CANVAS_SIZE), Image.Resampling.LANCZOS)

    st.markdown("""
    <style>
        [data-testid="stDialog"] [data-testid="stSelectbox"], [data-testid="stExpander"] [data-testid="stSelectbox"],
        [data-testid="stDialog"] [data-testid="stSelectbox"] > div, [data-testid="stExpander"] [data-testid="stSelectbox"] > div {
            width: 100% !important; max-width: 100% !important;
        }
        [data-testid="stDialog"] [data-testid="stMetric"] label, [data-testid="stDialog"] [data-testid="stMetric"] [data-testid="stMetricValue"],
        [data-testid="stExpander"] [data-testid="stMetric"] label, [data-testid="stExpander"] [data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-size: 0.95rem !important;
        }
        [data-testid="stDialog"] img, [data-testid="stExpander"] img { border-radius: 0 !important; }
    </style>
    """, unsafe_allow_html=True)

    if bf_img is not None:
        bf_resized = cv2.resize(bf_img, (CANVAS_SIZE, CANVAS_SIZE))
        bf_rgb = cv2.cvtColor(bf_resized, cv2.COLOR_GRAY2RGB) if bf_img.ndim == 2 else cv2.cvtColor(bf_resized, cv2.COLOR_BGR2RGB)
        left_col, right_col = st.columns(2, gap=None)
        with left_col:
            draw_mode = st.selectbox("Tool", DRAW_TOOLS, format_func=lambda x: TOOL_LABELS[x], key=f"draw_mode_region_{key_suffix}")
            st.caption("Left-click add, right-click close.  \nForce map (draw region)")
            canvas_result = st_canvas(
                fill_color="rgba(0, 188, 212, 0.25)", stroke_width=2, stroke_color="#00bcd4",
                background_image=pil_bg, drawing_mode=draw_mode, update_streamlit=True,
                height=CANVAS_SIZE, width=CANVAS_SIZE, display_toolbar=True,
                key=f"region_measure_canvas_{key_suffix}",
            )
        with right_col:
            vals = cell_vals if cell_vals else original_vals
            if vals:
                label = "Cell area" if cell_vals else "Full map"
                st.markdown(f'<p style="font-weight: 400; color: #334155; font-size: 0.95rem; margin: 0 20px 4px 4px;">{label}</p>', unsafe_allow_html=True)
                st.markdown(f"""
                <div style="width: 100%; box-sizing: border-box; border: 1px solid #e2e8f0; border-radius: 10px;
                    padding: 10px 12px; margin: 0 10px 20px 10px; background: linear-gradient(145deg, #f8fafc 0%, #f1f5f9 100%);
                    box-shadow: 0 1px 3px rgba(0,0,0,0.06);">
                    <div style="display: flex; flex-wrap: wrap; gap: 5px; font-size: 0.9rem;">
                        <span><strong>Sum:</strong> {vals['pixel_sum']:.1f}</span>
                        <span><strong>Force:</strong> {vals['force']:.1f}</span>
                        <span><strong>Max:</strong> {vals['max']:.3f}</span>
                        <span><strong>Mean:</strong> {vals['mean']:.3f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.caption("Bright-field")
            bf_display = bf_rgb.copy()
            if cell_mask is not None and np.any(cell_mask > 0):
                bf_display = _draw_contour_on_image(bf_display, cell_mask, stroke_color=(255, 0, 0), stroke_width=3)
            st.image(bf_display, width=CANVAS_SIZE)
    else:
        st.markdown("**Draw a region** on the heatmap.")
        draw_mode = st.selectbox("Drawing tool", DRAW_TOOLS,
            format_func=lambda x: "Polygon (free shape)" if x == "polygon" else TOOL_LABELS[x],
            key=f"draw_mode_region_{key_suffix}")
        st.caption("Polygon: left-click to add points, right-click to close.")
        canvas_result = st_canvas(
            fill_color="rgba(0, 188, 212, 0.25)", stroke_width=2, stroke_color="#00bcd4",
            background_image=pil_bg, drawing_mode=draw_mode, update_streamlit=True,
            height=CANVAS_SIZE, width=CANVAS_SIZE, display_toolbar=True,
            key=f"region_measure_canvas_{key_suffix}",
        )

    if canvas_result.json_data:
        masks = parse_canvas_shapes_to_masks(canvas_result.json_data, CANVAS_SIZE, CANVAS_SIZE, h, w)
        if masks:
            metrics_list = [compute_region_metrics(raw_heatmap, m, original_vals) for m in masks]
            if cell_mask is not None and np.any(cell_mask > 0):
                cell_metrics = compute_region_metrics(raw_heatmap, cell_mask, original_vals)
                metrics_list = [cell_metrics] + metrics_list
            render_region_metrics_and_downloads(
                metrics_list, masks, heatmap_rgb, input_filename, key_suffix, original_vals is not None,
                first_region_label="Auto boundary" if (cell_mask is not None and np.any(cell_mask > 0)) else None,
                bf_img=bf_img, cell_mask=cell_mask, colormap_name=colormap_name,
            )
