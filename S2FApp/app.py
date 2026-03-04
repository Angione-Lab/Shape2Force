"""
Shape2Force (S2F) - GUI for force map prediction from bright field microscopy images.
"""
import csv
import io
import os
import sys
import traceback

import cv2
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

import numpy as np
import streamlit as st
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots

S2F_ROOT = os.path.dirname(os.path.abspath(__file__))
if S2F_ROOT not in sys.path:
    sys.path.insert(0, S2F_ROOT)

from utils.substrate_settings import list_substrates

try:
    from streamlit_drawable_canvas import st_canvas
    HAS_DRAWABLE_CANVAS = True
except (ImportError, AttributeError):
    HAS_DRAWABLE_CANVAS = False

# Constants
MODEL_TYPE_LABELS = {"single_cell": "Single cell", "spheroid": "Spheroid"}
DRAW_TOOLS = ["polygon", "rect", "circle"]
TOOL_LABELS = {"polygon": "Polygon", "rect": "Rectangle", "circle": "Circle"}
CANVAS_SIZE = 320
SAMPLE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
CITATION = (
    "Lautaro Baro, Kaveh Shahhosseini, Amparo Andrés-Bordería, Claudio Angione, and Maria Angeles Juanes. "
    "**\"Shape-to-force (S2F): Predicting Cell Traction Forces from LabelFree Imaging\"**, 2026."
)


def _make_annotated_heatmap(heatmap_rgb, mask, fill_alpha=0.3, stroke_color=(255, 102, 0), stroke_width=2):
    """Composite heatmap with drawn region overlay. heatmap_rgb and mask must match in size."""
    annotated = heatmap_rgb.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Semi-transparent orange fill
    overlay = annotated.copy()
    cv2.fillPoly(overlay, contours, stroke_color)
    mask_3d = np.stack([mask] * 3, axis=-1).astype(bool)
    annotated[mask_3d] = (
        (1 - fill_alpha) * annotated[mask_3d].astype(np.float32)
        + fill_alpha * overlay[mask_3d].astype(np.float32)
    ).astype(np.uint8)
    # Orange contour
    cv2.drawContours(annotated, contours, -1, stroke_color, stroke_width)
    return annotated


def _parse_canvas_shapes_to_mask(json_data, canvas_h, canvas_w, heatmap_h, heatmap_w):
    """
    Parse drawn shapes from streamlit-drawable-canvas json_data and create a binary mask
    in heatmap coordinates. Returns (mask, num_shapes) or (None, 0) if no valid shapes.
    """
    if not json_data or "objects" not in json_data or not json_data["objects"]:
        return None, 0
    scale_x = heatmap_w / canvas_w
    scale_y = heatmap_h / canvas_h
    mask = np.zeros((heatmap_h, heatmap_w), dtype=np.uint8)
    count = 0
    for obj in json_data["objects"]:
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
                # Circle: (left, top) is mouse start point, not center.
                # Center = start + radius * (cos(angle), sin(angle))
                rx = ry = radius
                angle_rad = np.deg2rad(angle_deg)
                cx = left + radius * np.cos(angle_rad)
                cy = top + radius * np.sin(angle_rad)
            else:
                # Ellipse: left, top = top-left of bounding box
                rx = width / 2 if width > 0 else 0
                ry = height / 2 if height > 0 else 0
                if rx <= 0 or ry <= 0:
                    continue
                cx = left + rx
                cy = top + ry
            if rx <= 0 or ry <= 0:
                continue
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
                continue
            pts = np.array(pts, dtype=np.float32)
        else:
            continue
        pts[:, 0] *= scale_x
        pts[:, 1] *= scale_y
        pts = np.clip(pts, 0, [heatmap_w - 1, heatmap_h - 1]).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
        count += 1
    return mask, count


def _heatmap_to_png_bytes(scaled_heatmap):
    """Convert scaled heatmap (float 0-1) to PNG bytes buffer."""
    heatmap_uint8 = (np.clip(scaled_heatmap, 0, 1) * 255).astype(np.uint8)
    heatmap_rgb = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(heatmap_rgb).save(buf, format="PNG")
    buf.seek(0)
    return buf


def _build_original_vals(scaled_heatmap, pixel_sum, force, force_scale):
    """Build original_vals dict for measure tool."""
    return {
        "pixel_sum": pixel_sum * force_scale,
        "force": force * force_scale,
        "max": float(np.max(scaled_heatmap)),
        "mean": float(np.mean(scaled_heatmap)),
    }


def _render_result_display(img, scaled_heatmap, pixel_sum, force, force_scale, key_img, download_key_suffix=""):
    """Render prediction result: plot, metrics, expander, and download/measure buttons."""
    buf_hm = _heatmap_to_png_bytes(scaled_heatmap)
    base_name = os.path.splitext(key_img or "image")[0]
    main_csv_rows = [
        ["image", "Sum of all pixels", "Cell force (scaled)", "Heatmap max", "Heatmap mean"],
        [base_name, f"{pixel_sum * force_scale:.2f}", f"{force * force_scale:.2f}",
         f"{np.max(scaled_heatmap):.4f}", f"{np.mean(scaled_heatmap):.4f}"],
    ]
    buf_main_csv = io.StringIO()
    csv.writer(buf_main_csv).writerows(main_csv_rows)

    tit1, tit2 = st.columns(2)
    with tit1:
        st.markdown('<p style="font-size: 1.1rem; color: black; font-weight: 600;">Input: Bright-field image</p>', unsafe_allow_html=True)
    with tit2:
        st.markdown('<p style="font-size: 1.1rem; color: black; font-weight: 600;">Output: Predicted traction force map</p>', unsafe_allow_html=True)
    fig_pl = make_subplots(rows=1, cols=2)
    fig_pl.add_trace(go.Heatmap(z=img, colorscale="gray", showscale=False), row=1, col=1)
    fig_pl.add_trace(go.Heatmap(z=scaled_heatmap, colorscale="Jet", zmin=0, zmax=1, showscale=True,
        colorbar=dict(len=0.4, thickness=12)), row=1, col=2)
    fig_pl.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(scaleanchor="y", scaleratio=1),
        xaxis2=dict(scaleanchor="y2", scaleratio=1),
    )
    fig_pl.update_xaxes(showticklabels=False)
    fig_pl.update_yaxes(showticklabels=False, autorange="reversed")
    st.plotly_chart(fig_pl, use_container_width=True, config={"displayModeBar": True, "responsive": True})

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Sum of all pixels", f"{pixel_sum * force_scale:.2f}", help="Raw sum of all pixel values in the force map")
    with col2:
        st.metric("Cell force (scaled)", f"{force * force_scale:.2f}", help="Total traction force in physical units")
    with col3:
        st.metric("Heatmap max", f"{np.max(scaled_heatmap):.4f}", help="Peak force intensity in the map")
    with col4:
        st.metric("Heatmap mean", f"{np.mean(scaled_heatmap):.4f}", help="Average force intensity")

    with st.expander("How to read the results"):
        st.markdown("""
**Input (left):** Bright-field microscopy image of a cell or spheroid on a substrate.  
This is the raw image you provided—it shows cell shape but not forces.

**Output (right):** Predicted traction force map.  
- **Color** indicates force magnitude: blue = low, red = high  
- **Brighter/warmer colors** = stronger forces exerted by the cell on the substrate  
- Values are normalized to [0, 1] for visualization

**Metrics:**
- **Sum of all pixels:** Total force is the sum of all pixels in the force map. Each pixel represents the magnitude of force at that location; summing them gives the overall traction.
- **Cell force (scaled):** Total traction force in physical units (scaled by substrate stiffness)
- **Heatmap max/mean:** Peak and average force intensity in the map
        """)

    original_vals = _build_original_vals(scaled_heatmap, pixel_sum, force, force_scale)
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    with btn_col1:
        if HAS_DRAWABLE_CANVAS and st_dialog:
            if st.button("Measure tool", key="open_measure", icon=":material/straighten:"):
                st.session_state["open_measure_dialog"] = True
                st.rerun()
        elif HAS_DRAWABLE_CANVAS:
            with st.expander("Measure tool"):
                _render_region_canvas(
                    scaled_heatmap,
                    bf_img=img,
                    original_vals=original_vals,
                    key_suffix="expander",
                    input_filename=key_img,
                )
        else:
            st.caption("Install `streamlit-drawable-canvas-fix` for region measurement: `pip install streamlit-drawable-canvas-fix`")
    with btn_col2:
        st.download_button(
            "Download heatmap",
            width="stretch",
            data=buf_hm.getvalue(),
            file_name="s2f_heatmap.png",
            mime="image/png",
            key=f"download_heatmap{download_key_suffix}",
            icon=":material/download:",
        )
    with btn_col3:
        st.download_button(
            "Download values",
            width="stretch",
            data=buf_main_csv.getvalue(),
            file_name=f"{base_name}_main_values.csv",
            mime="text/csv",
            key=f"download_main_values{download_key_suffix}",
            icon=":material/download:",
        )


def _compute_region_metrics(scaled_heatmap, mask, original_vals=None):
    """Compute region metrics from mask. Returns dict with area_px, force_sum, density, etc."""
    area_px = int(np.sum(mask))
    region_values = scaled_heatmap * mask
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


def _render_region_metrics_and_downloads(metrics, heatmap_rgb, mask, input_filename, key_suffix, has_original_vals):
    """Render region metrics and download buttons."""
    base_name = os.path.splitext(input_filename or "image")[0]
    st.markdown("**Region (drawn)**")
    if has_original_vals:
        r1, r2, r3, r4, r5, r6 = st.columns(6)
        with r1:
            st.metric("Area", f"{metrics['area_px']:,}")
        with r2:
            st.metric("F.sum", f"{metrics['force_sum']:.3f}")
        with r3:
            st.metric("Force", f"{metrics['force_scaled']:.1f}")
        with r4:
            st.metric("Max", f"{metrics['max']:.3f}")
        with r5:
            st.metric("Mean", f"{metrics['mean']:.3f}")
        with r6:
            st.metric("Density", f"{metrics['density']:.4f}")
        csv_rows = [
            ["image", "Area", "F.sum", "Force", "Max", "Mean", "Density"],
            [base_name, metrics["area_px"], f"{metrics['force_sum']:.3f}", f"{metrics['force_scaled']:.1f}",
             f"{metrics['max']:.3f}", f"{metrics['mean']:.3f}", f"{metrics['density']:.4f}"],
        ]
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Area (px²)", f"{metrics['area_px']:,}")
        with c2:
            st.metric("Force sum", f"{metrics['force_sum']:.4f}")
        with c3:
            st.metric("Density", f"{metrics['density']:.6f}")
        csv_rows = [
            ["image", "Area", "Force sum", "Density"],
            [base_name, metrics["area_px"], f"{metrics['force_sum']:.4f}", f"{metrics['density']:.6f}"],
        ]
    buf_csv = io.StringIO()
    csv.writer(buf_csv).writerows(csv_rows)
    buf_img = io.BytesIO()
    Image.fromarray(_make_annotated_heatmap(heatmap_rgb, mask)).save(buf_img, format="PNG")
    buf_img.seek(0)
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button("Download values", data=buf_csv.getvalue(),
            file_name=f"{base_name}_region_values.csv", mime="text/csv",
            key=f"download_region_values_{key_suffix}", icon=":material/download:")
    with dl_col2:
        st.download_button("Download annotated heatmap", data=buf_img.getvalue(),
            file_name=f"{base_name}_annotated_heatmap.png", mime="image/png",
            key=f"download_annotated_{key_suffix}", icon=":material/image:")


def _render_region_canvas(scaled_heatmap, bf_img=None, original_vals=None, key_suffix="", input_filename=None):
    """Render drawable canvas and region metrics. Used in dialog or expander."""
    h, w = scaled_heatmap.shape
    heatmap_display = (np.clip(scaled_heatmap, 0, 1) * 255).astype(np.uint8)
    heatmap_rgb = cv2.cvtColor(cv2.applyColorMap(heatmap_display, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
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
                fill_color="rgba(255, 165, 0, 0.3)", stroke_width=2, stroke_color="#ff6600",
                background_image=pil_bg, drawing_mode=draw_mode, update_streamlit=True,
                height=CANVAS_SIZE, width=CANVAS_SIZE, display_toolbar=True,
                key=f"region_measure_canvas_{key_suffix}",
            )
        with right_col:
            if original_vals:
                st.markdown('<p style="font-weight: 400; color: #334155; font-size: 0.95rem; margin: 0 20px 4px 4px;">Full map</p>', unsafe_allow_html=True)
                st.markdown(f"""
                <div style="width: 100%; box-sizing: border-box; border: 1px solid #e2e8f0; border-radius: 10px;
                    padding: 10px 12px; margin: 0 10px 20px 10px; background: linear-gradient(145deg, #f8fafc 0%, #f1f5f9 100%);
                    box-shadow: 0 1px 3px rgba(0,0,0,0.06);">
                    <div style="display: flex; flex-wrap: wrap; gap: 5px; font-size: 0.9rem;">
                        <span><strong>Sum:</strong> {original_vals['pixel_sum']:.1f}</span>
                        <span><strong>Force:</strong> {original_vals['force']:.1f}</span>
                        <span><strong>Max:</strong> {original_vals['max']:.3f}</span>
                        <span><strong>Mean:</strong> {original_vals['mean']:.3f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.caption("Bright-field")
            st.image(bf_rgb, width=CANVAS_SIZE)
    else:
        st.markdown("**Draw a region** on the heatmap.")
        draw_mode = st.selectbox("Drawing tool", DRAW_TOOLS,
            format_func=lambda x: "Polygon (free shape)" if x == "polygon" else TOOL_LABELS[x],
            key=f"draw_mode_region_{key_suffix}")
        st.caption("Polygon: left-click to add points, right-click to close.")
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)", stroke_width=2, stroke_color="#ff6600",
            background_image=pil_bg, drawing_mode=draw_mode, update_streamlit=True,
            height=CANVAS_SIZE, width=CANVAS_SIZE, display_toolbar=True,
            key=f"region_measure_canvas_{key_suffix}",
        )

    if canvas_result.json_data:
        mask, n = _parse_canvas_shapes_to_mask(canvas_result.json_data, CANVAS_SIZE, CANVAS_SIZE, h, w)
        if mask is not None and n > 0:
            metrics = _compute_region_metrics(scaled_heatmap, mask, original_vals)
            _render_region_metrics_and_downloads(metrics, heatmap_rgb, mask, input_filename, key_suffix, original_vals is not None)


st_dialog = getattr(st, "dialog", None) or getattr(st, "experimental_dialog", None)
if HAS_DRAWABLE_CANVAS and st_dialog:
    @st_dialog("Measure tool", width="medium")
    def measure_region_dialog():
        scaled_heatmap = st.session_state.get("measure_scaled_heatmap")
        if scaled_heatmap is None:
            st.warning("No prediction available to measure.")
            return
        bf_img = st.session_state.get("measure_bf_img")
        original_vals = st.session_state.get("measure_original_vals")
        input_filename = st.session_state.get("measure_input_filename", "image")
        _render_region_canvas(scaled_heatmap, bf_img=bf_img, original_vals=original_vals, key_suffix="dialog", input_filename=input_filename)
else:
    def measure_region_dialog():
        pass  # no-op when canvas or dialog not available


st.set_page_config(page_title="Shape2Force (S2F)", page_icon="🦠", layout="centered")
st.markdown("""
<style>
section[data-testid="stSidebar"] { width: 380px !important; }
div[data-testid="stHorizontalBlock"]:has([data-testid="stDownloadButton"]):has([data-testid="stButton"]) > div {
    flex: 1 1 0 !important; min-width: 0 !important;
}
div[data-testid="stHorizontalBlock"]:has([data-testid="stDownloadButton"]):has([data-testid="stButton"]) button {
    width: 100% !important; min-width: 100px !important; white-space: nowrap !important;
}
div[data-testid="stHorizontalBlock"]:has([data-testid="stDownloadButton"]):has([data-testid="stButton"]) > div:nth-child(1) button {
    background-color: #0d9488 !important; color: white !important; border-color: #0d9488 !important;
}
div[data-testid="stHorizontalBlock"]:has([data-testid="stDownloadButton"]):has([data-testid="stButton"]) > div:nth-child(1) button:hover {
    background-color: #0f766e !important; border-color: #0f766e !important; color: white !important;
}
</style>
""", unsafe_allow_html=True)
st.title("🦠 Shape2Force (S2F)")
st.caption("Predict traction force maps from bright-field microscopy images of cells or spheroids")

# Folders: checkpoints in subfolders by model type (single_cell / spheroid)
ckp_base = os.path.join(S2F_ROOT, "ckp")
# Fallback: use project root ckp when running from S2F repo (ckp at S2F/ckp/)
if not os.path.isdir(ckp_base):
    project_root = os.path.dirname(S2F_ROOT)
    if os.path.isdir(os.path.join(project_root, "ckp")):
        ckp_base = os.path.join(project_root, "ckp")
ckp_single_cell = os.path.join(ckp_base, "single_cell")
ckp_spheroid = os.path.join(ckp_base, "spheroid")
sample_base = os.path.join(S2F_ROOT, "samples")
sample_single_cell = os.path.join(sample_base, "single_cell")
sample_spheroid = os.path.join(sample_base, "spheroid")


def get_ckp_files_for_model(model_type):
    """Return list of .pth files in the checkpoint folder for the given model type."""
    folder = ckp_single_cell if model_type == "single_cell" else ckp_spheroid
    if os.path.isdir(folder):
        return sorted(f for f in os.listdir(folder) if f.endswith(".pth"))
    return []


def get_sample_files_for_model(model_type):
    """Return list of sample images in the sample folder for the given model type."""
    folder = sample_single_cell if model_type == "single_cell" else sample_spheroid
    if os.path.isdir(folder):
        return sorted(f for f in os.listdir(folder) if f.lower().endswith(SAMPLE_EXTENSIONS))
    return []

# Sidebar: model configuration
with st.sidebar:
    st.header("Model configuration")
    model_type = st.radio(
        "Model type",
        ["single_cell", "spheroid"],
        format_func=lambda x: MODEL_TYPE_LABELS[x],
        horizontal=False,
        help="Single cell: substrate-aware force prediction. Spheroid: spheroid force maps.",
    )
    st.caption(f"Inference mode: **{MODEL_TYPE_LABELS[model_type]}**")

    ckp_files = get_ckp_files_for_model(model_type)
    ckp_folder = ckp_single_cell if model_type == "single_cell" else ckp_spheroid
    ckp_subfolder_name = "single_cell" if model_type == "single_cell" else "spheroid"

    if ckp_files:
        checkpoint = st.selectbox(
            "Checkpoint",
            ckp_files,
            help=f"Select a .pth file from ckp/{ckp_subfolder_name}/",
        )
    else:
        st.warning(f"No .pth files in ckp/{ckp_subfolder_name}/. Add checkpoints to load.")
        checkpoint = None

    substrate_config = None
    substrate_val = "fibroblasts_PDMS"
    use_manual = False
    if model_type == "single_cell":
        try:
            substrates = list_substrates()
            substrate_val = st.selectbox(
                "Substrate (from config)",
                substrates,
                help="Select a preset from config/substrate_settings.json",
            )
            use_manual = st.checkbox("Enter substrate values manually", value=False)
            if use_manual:
                st.caption("Enter pixelsize (µm/px) and Young's modulus (Pa)")
                manual_pixelsize = st.number_input("Pixelsize (µm/px)", min_value=0.1, max_value=50.0,
                                                   value=3.0769, step=0.1, format="%.4f")
                manual_young = st.number_input("Young's modulus (Pa)", min_value=100.0, max_value=100000.0,
                                               value=6000.0, step=100.0, format="%.0f")
                substrate_config = {"pixelsize": manual_pixelsize, "young": manual_young}
        except FileNotFoundError:
            st.error("config/substrate_settings.json not found")

    st.divider()
    st.header("Display options")
    force_scale = st.slider(
        "Force scale",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.01,
        format="%.2f",
        help="Scale the displayed force values. 1 = full intensity, 0.5 = half the pixel values.",
    )

# Main area: image input
img_source = st.radio("Image source", ["Upload", "Example"], horizontal=True, label_visibility="collapsed")
img = None
uploaded = None
selected_sample = None

if img_source == "Upload":
    uploaded = st.file_uploader(
        "Upload bright-field image",
        type=["tif", "tiff", "png", "jpg", "jpeg"],
        help="Bright-field microscopy image of a cell or spheroid on a substrate (grayscale or RGB). The model will predict traction forces from the cell shape.",
    )
    if uploaded:
        bytes_data = uploaded.read()
        nparr = np.frombuffer(bytes_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        uploaded.seek(0)  # reset for potential re-read
else:
    sample_files = get_sample_files_for_model(model_type)
    sample_folder = sample_single_cell if model_type == "single_cell" else sample_spheroid
    sample_subfolder_name = "single_cell" if model_type == "single_cell" else "spheroid"
    if sample_files:
        selected_sample = st.selectbox(
            f"Select example image (from `samples/{sample_subfolder_name}/`)",
            sample_files,
            format_func=lambda x: x,
            key=f"sample_{model_type}",
        )
        if selected_sample:
            sample_path = os.path.join(sample_folder, selected_sample)
            img = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
        # Show example thumbnails (filtered by model type)
        n_cols = min(5, len(sample_files))
        cols = st.columns(n_cols)
        for i, fname in enumerate(sample_files[:8]):  # show up to 8
            with cols[i % n_cols]:
                path = os.path.join(sample_folder, fname)
                sample_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if sample_img is not None:
                    st.image(sample_img, caption=fname, width='content')
    else:
        st.info(f"No example images in samples/{sample_subfolder_name}/. Add images or use Upload.")

col_btn, col_model, col_path = st.columns([1, 1, 3])
with col_btn:
    run = st.button("Run prediction", type="primary")
with col_model:
    st.markdown(f"<span style='display: inline-flex; align-items: center; height: 38px;'>{MODEL_TYPE_LABELS[model_type]}</span>", unsafe_allow_html=True)
with col_path:
    ckp_path = f"ckp/{ckp_subfolder_name}/{checkpoint}" if checkpoint else f"ckp/{ckp_subfolder_name}/"
    st.markdown(f"<span style='display: inline-flex; align-items: center; height: 38px;'>Checkpoint: <code>{ckp_path}</code></span>", unsafe_allow_html=True)
has_image = img is not None

# Persist results in session state so they survive re-runs (e.g. when clicking Download)
if "prediction_result" not in st.session_state:
    st.session_state["prediction_result"] = None

# Show results if we just ran prediction OR we have cached results from a previous run
just_ran = run and checkpoint and has_image
cached = st.session_state["prediction_result"]
key_img = (uploaded.name if uploaded else None) if img_source == "Upload" else selected_sample
current_key = (model_type, checkpoint, key_img)
has_cached = cached is not None and cached.get("cache_key") == current_key

if just_ran:
    st.session_state["prediction_result"] = None  # Clear before new run
    with st.spinner("Loading model and predicting..."):
        try:
            from predictor import S2FPredictor
            predictor = S2FPredictor(
                model_type=model_type,
                checkpoint_path=checkpoint,
                ckp_folder=ckp_folder,
            )
            if img is not None:
                sub_val = substrate_val if model_type == "single_cell" and not use_manual else "fibroblasts_PDMS"
                heatmap, force, pixel_sum = predictor.predict(
                    image_array=img,
                    substrate=sub_val,
                    substrate_config=substrate_config if model_type == "single_cell" else None,
                )

                st.success("Prediction complete!")

                scaled_heatmap = heatmap * force_scale

                # Store result and measure data before rendering (Measure click survives rerun)
                cache_key = (model_type, checkpoint, key_img)
                st.session_state["prediction_result"] = {
                    "img": img.copy(),
                    "heatmap": heatmap.copy(),
                    "force": force,
                    "pixel_sum": pixel_sum,
                    "cache_key": cache_key,
                }
                st.session_state["measure_scaled_heatmap"] = scaled_heatmap.copy()
                st.session_state["measure_bf_img"] = img.copy()
                st.session_state["measure_input_filename"] = key_img or "image"
                st.session_state["measure_original_vals"] = _build_original_vals(scaled_heatmap, pixel_sum, force, force_scale)

                _render_result_display(img, scaled_heatmap, pixel_sum, force, force_scale, key_img)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.code(traceback.format_exc())

elif has_cached:
    r = st.session_state["prediction_result"]
    img, heatmap, force, pixel_sum = r["img"], r["heatmap"], r["force"], r["pixel_sum"]
    scaled_heatmap = heatmap * force_scale

    st.session_state["measure_scaled_heatmap"] = scaled_heatmap.copy()
    st.session_state["measure_bf_img"] = img.copy()
    st.session_state["measure_input_filename"] = key_img or "image"
    st.session_state["measure_original_vals"] = _build_original_vals(scaled_heatmap, pixel_sum, force, force_scale)

    if st.session_state.pop("open_measure_dialog", False):
        measure_region_dialog()

    st.success("Prediction complete!")
    _render_result_display(img, scaled_heatmap, pixel_sum, force, force_scale, key_img, download_key_suffix="_cached")

elif run and not checkpoint:
    st.warning("Please add checkpoint files to the ckp/ folder and select one.")
elif run and not has_image:
    st.warning("Please upload an image or select an example.")

st.sidebar.divider()
st.sidebar.caption(f"Examples: `samples/{ckp_subfolder_name}/`")
st.sidebar.caption("If you find this software useful, please cite:")
st.sidebar.caption(CITATION)
