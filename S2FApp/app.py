"""
Shape2Force (S2F) - GUI for force map prediction from bright field microscopy images.
"""
import os
import sys
import traceback

import cv2
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

import numpy as np
import streamlit as st

S2F_ROOT = os.path.dirname(os.path.abspath(__file__))
if S2F_ROOT not in sys.path:
    sys.path.insert(0, S2F_ROOT)

from config.constants import (
    COLORMAPS,
    MODEL_TYPE_LABELS,
    SAMPLE_EXTENSIONS,
)
from utils.paths import get_ckp_base, model_subfolder
from utils.segmentation import estimate_cell_mask
from utils.substrate_settings import list_substrates
from utils.display import apply_display_scale
from ui.components import (
    build_original_vals,
    build_cell_vals,
    render_result_display,
    render_region_canvas,
    ST_DIALOG,
    HAS_DRAWABLE_CANVAS,
)

CITATION = (
    "Lautaro Baro, Kaveh Shahhosseini, Amparo Andrés-Bordería, Claudio Angione, and Maria Angeles Juanes. "
    "**\"Shape-to-force (S2F): Predicting Cell Traction Forces from LabelFree Imaging\"**, 2026."
)

# Measure tool dialog: defined early so it exists before render_result_display uses it
if HAS_DRAWABLE_CANVAS and ST_DIALOG:
    @ST_DIALOG("Measure tool", width="medium")
    def measure_region_dialog():
        raw_heatmap = st.session_state.get("measure_raw_heatmap")
        if raw_heatmap is None:
            st.warning("No prediction available to measure.")
            return
        display_mode = st.session_state.get("measure_display_mode", "Auto")
        display_heatmap = apply_display_scale(raw_heatmap, display_mode)
        bf_img = st.session_state.get("measure_bf_img")
        original_vals = st.session_state.get("measure_original_vals")
        cell_vals = st.session_state.get("measure_cell_vals")
        cell_mask = st.session_state.get("measure_cell_mask")
        input_filename = st.session_state.get("measure_input_filename", "image")
        colormap_name = st.session_state.get("measure_colormap", "Jet")
        render_region_canvas(
            display_heatmap, raw_heatmap=raw_heatmap, bf_img=bf_img,
            original_vals=original_vals, cell_vals=cell_vals, cell_mask=cell_mask,
            key_suffix="dialog", input_filename=input_filename, colormap_name=colormap_name,
        )
else:
    def measure_region_dialog():
        pass


def _get_measure_dialog_fn():
    """Return measure dialog callable if available, else None (fixes st_dialog ordering)."""
    return measure_region_dialog if (HAS_DRAWABLE_CANVAS and ST_DIALOG) else None


def _populate_measure_session_state(heatmap, img, pixel_sum, force, key_img, colormap_name,
                                    display_mode, auto_cell_boundary):
    """Populate session state for the measure tool."""
    cell_mask = estimate_cell_mask(heatmap)
    st.session_state["measure_raw_heatmap"] = heatmap.copy()
    st.session_state["measure_display_mode"] = display_mode
    st.session_state["measure_bf_img"] = img.copy()
    st.session_state["measure_input_filename"] = key_img or "image"
    st.session_state["measure_original_vals"] = build_original_vals(heatmap, pixel_sum, force)
    st.session_state["measure_colormap"] = colormap_name
    st.session_state["measure_auto_cell_on"] = auto_cell_boundary
    st.session_state["measure_cell_vals"] = build_cell_vals(heatmap, cell_mask, pixel_sum, force) if auto_cell_boundary else None
    st.session_state["measure_cell_mask"] = cell_mask if auto_cell_boundary else None


st.set_page_config(page_title="Shape2Force (S2F)", page_icon="🦠", layout="centered")

# Theme CSS (inject based on sidebar selection)
def _inject_theme_css(theme):
    if theme == "Dark":
        st.markdown("""
        <style>
        .stApp { background-color: #0e1117 !important; }
        .stApp header { background-color: #0e1117 !important; }
        section[data-testid="stSidebar"] { background-color: #1a1a2e !important; }
        section[data-testid="stSidebar"] .stMarkdown { color: #fafafa !important; }
        section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] { color: #e2e8f0 !important; }
        h1, h2, h3 { color: #fafafa !important; }
        p { color: #e2e8f0 !important; }
        .stCaption { color: #94a3b8 !important; }
        </style>
        """, unsafe_allow_html=True)


st.markdown("""
<style>
section[data-testid="stSidebar"] { width: 380px !important; }
section[data-testid="stSidebar"] h2 {
  font-size: 1.25rem !important;
  font-weight: 600 !important;
}
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
  font-size: 0.95rem !important;
  font-weight: 500 !important;
}
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

# Folders
ckp_base = get_ckp_base(S2F_ROOT)
ckp_single_cell = os.path.join(ckp_base, "single_cell")
ckp_spheroid = os.path.join(ckp_base, "spheroid")
sample_base = os.path.join(S2F_ROOT, "samples")
sample_single_cell = os.path.join(sample_base, "single_cell")
sample_spheroid = os.path.join(sample_base, "spheroid")


def get_ckp_files_for_model(model_type):
    folder = ckp_single_cell if model_type == "single_cell" else ckp_spheroid
    if os.path.isdir(folder):
        return sorted(f for f in os.listdir(folder) if f.endswith(".pth"))
    return []


def get_sample_files_for_model(model_type):
    folder = sample_single_cell if model_type == "single_cell" else sample_spheroid
    if os.path.isdir(folder):
        return sorted(f for f in os.listdir(folder) if f.lower().endswith(SAMPLE_EXTENSIONS))
    return []


def get_cached_sample_thumbnails(model_type, sample_folder, sample_files):
    """Return cached sample thumbnails. Key by (model_type, tuple(files))."""
    cache_key = (model_type, tuple(sample_files))
    if "sample_thumbnails" not in st.session_state:
        st.session_state["sample_thumbnails"] = {}
    cache = st.session_state["sample_thumbnails"]
    if cache_key not in cache:
        thumbnails = []
        for fname in sample_files[:8]:
            path = os.path.join(sample_folder, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            thumbnails.append((fname, img))
        cache[cache_key] = thumbnails
    return cache[cache_key]


# Sidebar
with st.sidebar:
    st.header("Settings")

    model_type = st.radio(
        "Model type",
        ["single_cell", "spheroid"],
        format_func=lambda x: MODEL_TYPE_LABELS[x],
        horizontal=False,
        help="Single cell: substrate-aware force prediction. Spheroid: spheroid force maps.",
    )

    ckp_files = get_ckp_files_for_model(model_type)
    ckp_folder = ckp_single_cell if model_type == "single_cell" else ckp_spheroid
    ckp_subfolder_name = model_subfolder(model_type)

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
    substrate_val = "Fibroblasts_Fibronectin_6KPa"
    use_manual = True
    if model_type == "single_cell":
        try:
            st.markdown('<p style="font-size: 0.95rem; font-weight: 500; margin-bottom: 0.5rem;">Conditions</p>', unsafe_allow_html=True)
            conditions_source = st.radio(
                "Conditions",
                ["Manually", "From config"],
                horizontal=True,
                label_visibility="collapsed",
            )
            from_config = conditions_source == "From config"
            if from_config:
                substrate_config = None
                substrates = list_substrates()
                substrate_val = st.selectbox(
                    "Conditions (from config)",
                    substrates,
                    help="Select a preset from config/substrate_settings.json",
                    label_visibility="collapsed",
                )
                use_manual = False
            else:
                manual_pixelsize = st.number_input("Pixel size (µm/px)", min_value=0.1, max_value=50.0,
                                                   value=3.0769, step=0.1, format="%.4f")
                manual_young = st.number_input("Pascals", min_value=100.0, max_value=100000.0,
                                               value=6000.0, step=100.0, format="%.0f")
                substrate_config = {"pixelsize": manual_pixelsize, "young": manual_young}
                use_manual = True
        except FileNotFoundError:
            st.error("config/substrate_settings.json not found")

    display_mode = st.radio(
        "Force scale",
        ["Auto", "Fixed"],
        help="Auto: map data range to full color scale (Fiji-style). Fixed: use 0-1 range. Metrics always show raw values.",
        horizontal=True,
    )
    colormap_name = st.selectbox(
        "Heatmap colormap",
        list(COLORMAPS.keys()),
        help="Color scheme for the force map. Viridis is often preferred for accessibility.",
    )

    auto_cell_boundary = st.checkbox(
        "Auto boundary",
        value=True,
        help="When on: estimate cell region from force map and use it for metrics (red contour). When off: use entire map.",
    )

    theme = st.radio("Theme", ["Light", "Dark"], horizontal=True, key="theme_selector")
    _inject_theme_css(theme)

# Main area: image input
img_source = st.radio("Image source", ["Upload", "Example"], horizontal=True, label_visibility="collapsed")
img = None
uploaded = None
selected_sample = None

if img_source == "Upload":
    uploaded = st.file_uploader(
        "Upload bright-field image",
        type=["tif", "tiff", "png", "jpg", "jpeg"],
        help="Bright-field microscopy image of a cell or spheroid on a substrate (grayscale or RGB).",
    )
    if uploaded:
        bytes_data = uploaded.read()
        nparr = np.frombuffer(bytes_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        uploaded.seek(0)
else:
    sample_files = get_sample_files_for_model(model_type)
    sample_folder = sample_single_cell if model_type == "single_cell" else sample_spheroid
    sample_subfolder_name = model_subfolder(model_type)
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
        # Cached thumbnails
        thumbnails = get_cached_sample_thumbnails(model_type, sample_folder, sample_files)
        n_cols = min(5, len(thumbnails))
        cols = st.columns(n_cols)
        for i, (fname, sample_img) in enumerate(thumbnails):
            if sample_img is not None:
                with cols[i % n_cols]:
                    st.image(sample_img, caption=fname, width=120)
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

if "prediction_result" not in st.session_state:
    st.session_state["prediction_result"] = None

just_ran = run and checkpoint and has_image
cached = st.session_state["prediction_result"]
key_img = (uploaded.name if uploaded else None) if img_source == "Upload" else selected_sample
current_key = (model_type, checkpoint, key_img)
has_cached = cached is not None and cached.get("cache_key") == current_key


def get_or_create_predictor(model_type, checkpoint, ckp_folder):
    """Cache predictor in session state. Invalidate when model/checkpoint changes."""
    cache_key = (model_type, checkpoint)
    if "predictor" not in st.session_state or st.session_state.get("predictor_key") != cache_key:
        from predictor import S2FPredictor
        st.session_state["predictor"] = S2FPredictor(
            model_type=model_type,
            checkpoint_path=checkpoint,
            ckp_folder=ckp_folder,
        )
        st.session_state["predictor_key"] = cache_key
    return st.session_state["predictor"]


if just_ran:
    st.session_state["prediction_result"] = None
    with st.spinner("Loading model and predicting..."):
        try:
            predictor = get_or_create_predictor(model_type, checkpoint, ckp_folder)
            sub_val = substrate_val if model_type == "single_cell" and not use_manual else "Fibroblasts_Fibronectin_6KPa"
            heatmap, force, pixel_sum = predictor.predict(
                image_array=img,
                substrate=sub_val,
                substrate_config=substrate_config if model_type == "single_cell" else None,
            )

            st.success("Prediction complete!")

            display_heatmap = apply_display_scale(heatmap, display_mode)

            cache_key = (model_type, checkpoint, key_img)
            st.session_state["prediction_result"] = {
                "img": img.copy(),
                "heatmap": heatmap.copy(),
                "force": force,
                "pixel_sum": pixel_sum,
                "cache_key": cache_key,
            }
            _populate_measure_session_state(
                heatmap, img, pixel_sum, force, key_img, colormap_name,
                display_mode, auto_cell_boundary,
            )

            render_result_display(
                img, heatmap, display_heatmap, pixel_sum, force, key_img,
                colormap_name=colormap_name,
                display_mode=display_mode,
                measure_region_dialog=_get_measure_dialog_fn(),
                auto_cell_boundary=auto_cell_boundary,
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.code(traceback.format_exc())

elif has_cached:
    r = st.session_state["prediction_result"]
    img, heatmap, force, pixel_sum = r["img"], r["heatmap"], r["force"], r["pixel_sum"]
    display_heatmap = apply_display_scale(heatmap, display_mode)

    _populate_measure_session_state(
        heatmap, img, pixel_sum, force, key_img, colormap_name,
        display_mode, auto_cell_boundary,
    )

    if st.session_state.pop("open_measure_dialog", False):
        measure_region_dialog()

    st.success("Prediction complete!")
    render_result_display(
        img, heatmap, display_heatmap, pixel_sum, force, key_img,
        download_key_suffix="_cached",
        colormap_name=colormap_name,
        display_mode=display_mode,
        measure_region_dialog=_get_measure_dialog_fn(),
        auto_cell_boundary=auto_cell_boundary,
    )

elif run and not checkpoint:
    st.warning("Please add checkpoint files to the ckp/ folder and select one.")
elif run and not has_image:
    st.warning("Please upload an image or select an example.")

st.sidebar.divider()
st.sidebar.caption("If you find this software useful, please cite:")
st.sidebar.caption(CITATION)
