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
    DEFAULT_SUBSTRATE,
    MODEL_TYPE_LABELS,
    SAMPLE_EXTENSIONS,
    SAMPLE_THUMBNAIL_LIMIT,
)
from utils.paths import get_ckp_base, get_ckp_folder, get_sample_folder, list_files_in_folder, model_subfolder
from utils.segmentation import estimate_cell_mask
from utils.substrate_settings import list_substrates
from utils.display import apply_display_scale
from ui.components import (
    build_original_vals,
    build_cell_vals,
    render_result_display,
    render_region_canvas,
    render_system_status,
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
        display_mode = st.session_state.get("measure_display_mode", "Full")
        display_heatmap = apply_display_scale(
            raw_heatmap, display_mode,
            min_percentile=st.session_state.get("measure_min_percentile", 0),
            max_percentile=st.session_state.get("measure_max_percentile", 100),
            clip_min=st.session_state.get("measure_clip_min", 0),
            clip_max=st.session_state.get("measure_clip_max", 1),
        )
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
                                    display_mode, auto_cell_boundary, cell_mask=None,
                                    min_percentile=0, max_percentile=100, clip_min=0, clip_max=1):
    """Populate session state for the measure tool. If cell_mask is None and auto_cell_boundary, computes it."""
    if cell_mask is None and auto_cell_boundary:
        cell_mask = estimate_cell_mask(heatmap)
    st.session_state["measure_raw_heatmap"] = heatmap.copy()
    st.session_state["measure_display_mode"] = display_mode
    st.session_state["measure_min_percentile"] = min_percentile
    st.session_state["measure_max_percentile"] = max_percentile
    st.session_state["measure_clip_min"] = clip_min
    st.session_state["measure_clip_max"] = clip_max
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


def get_cached_sample_thumbnails(model_type, sample_folder, sample_files):
    """Return cached sample thumbnails. Key by (model_type, tuple(files))."""
    cache_key = (model_type, tuple(sample_files))
    if "sample_thumbnails" not in st.session_state:
        st.session_state["sample_thumbnails"] = {}
    cache = st.session_state["sample_thumbnails"]
    if cache_key not in cache:
        thumbnails = []
        for fname in sample_files[:SAMPLE_THUMBNAIL_LIMIT]:
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

    ckp_folder = get_ckp_folder(ckp_base, model_type)
    ckp_files = list_files_in_folder(ckp_folder, ".pth")
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
    substrate_val = DEFAULT_SUBSTRATE
    use_manual = True
    if model_type == "single_cell":
        try:
            st.markdown('<p style="font-size: 0.95rem; font-weight: 500; margin-bottom: 0.5rem;">Conditions</p>', unsafe_allow_html=True)
            conditions_source = st.radio(
                "Conditions",
                ["From config", "Manually"],
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

    auto_cell_boundary = st.toggle(
        "Auto boundary",
        value=False,
        help="When on: estimate cell region from force map and use it for metrics (red contour). When off: use entire map.",
    )

    st.markdown('<p style="font-size: 0.95rem; font-weight: 500; margin-bottom: 0.5rem;">Heatmap display</p>', unsafe_allow_html=True)
    display_mode = st.radio(
        "Mode",
        ["Full", "Percentile", "Rescale", "Clip", "Filter"],
        help="Full: 0–1 as-is. Percentile: min/max percentiles. Rescale: stretch range to colors. Clip: clip, keep scale. Filter: show only in range.",
        horizontal=True,
        label_visibility="collapsed",
    )
    min_percentile, max_percentile = 0, 100
    clip_min, clip_max = 0.0, 1.0
    if display_mode == "Percentile":
        col_pmin, col_pmax = st.columns(2)
        with col_pmin:
            min_percentile = st.slider("Min percentile", 0, 100, 2, 1, help="Values below this percentile → black")
        with col_pmax:
            max_percentile = st.slider("Max percentile", 0, 100, 99, 1, help="Values above this percentile → white")
        if min_percentile >= max_percentile:
            st.warning("Min percentile must be less than max. Using min=0, max=100.")
            min_percentile, max_percentile = 0, 100
    elif display_mode in ("Rescale", "Clip", "Filter"):
        col_cmin, col_cmax = st.columns(2)
        with col_cmin:
            clip_min = st.number_input("Min", value=0.0, min_value=None, max_value=None, step=0.01, format="%.3f",
                                       help="Rescale: below → black. Clip: clamp to min. Filter: below → discarded.")
        with col_cmax:
            clip_max = st.number_input("Max", value=1.0, min_value=None, max_value=None, step=0.01, format="%.3f",
                                       help="Rescale: above → white. Clip: clamp to max. Filter: above → discarded.")
        if clip_min >= clip_max:
            st.warning("Min must be less than max. Using min=0, max=1.")
            clip_min, clip_max = 0.0, 1.0
    colormap_name = st.selectbox(
        "Heatmap colormap",
        list(COLORMAPS.keys()),
        help="Color scheme for the force map. Viridis is often preferred for accessibility.",
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
    sample_folder = get_sample_folder(S2F_ROOT, model_type)
    sample_files = list_files_in_folder(sample_folder, SAMPLE_EXTENSIONS)
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
            sub_val = substrate_val if model_type == "single_cell" and not use_manual else DEFAULT_SUBSTRATE
            heatmap, force, pixel_sum = predictor.predict(
                image_array=img,
                substrate=sub_val,
                substrate_config=substrate_config if model_type == "single_cell" else None,
            )

            st.success("Prediction complete!")

            display_heatmap = apply_display_scale(
                heatmap, display_mode,
                min_percentile=min_percentile,
                max_percentile=max_percentile,
                clip_min=clip_min,
                clip_max=clip_max,
            )

            cache_key = (model_type, checkpoint, key_img)
            st.session_state["prediction_result"] = {
                "img": img.copy(),
                "heatmap": heatmap.copy(),
                "force": force,
                "pixel_sum": pixel_sum,
                "cache_key": cache_key,
            }
            cell_mask = estimate_cell_mask(heatmap) if auto_cell_boundary else None
            _populate_measure_session_state(
                heatmap, img, pixel_sum, force, key_img, colormap_name,
                display_mode, auto_cell_boundary, cell_mask=cell_mask,
                min_percentile=min_percentile, max_percentile=max_percentile,
                clip_min=clip_min, clip_max=clip_max,
            )
            render_result_display(
                img, heatmap, display_heatmap, pixel_sum, force, key_img,
                colormap_name=colormap_name,
                display_mode=display_mode,
                measure_region_dialog=_get_measure_dialog_fn(),
                auto_cell_boundary=auto_cell_boundary,
                cell_mask=cell_mask,
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.code(traceback.format_exc())

elif has_cached:
    r = st.session_state["prediction_result"]
    img, heatmap, force, pixel_sum = r["img"], r["heatmap"], r["force"], r["pixel_sum"]
    display_heatmap = apply_display_scale(
        heatmap, display_mode,
        min_percentile=min_percentile,
        max_percentile=max_percentile,
        clip_min=clip_min,
        clip_max=clip_max,
    )
    cell_mask = estimate_cell_mask(heatmap) if auto_cell_boundary else None
    _populate_measure_session_state(
        heatmap, img, pixel_sum, force, key_img, colormap_name,
        display_mode, auto_cell_boundary, cell_mask=cell_mask,
        min_percentile=min_percentile, max_percentile=max_percentile,
        clip_min=clip_min, clip_max=clip_max,
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
        cell_mask=cell_mask,
    )

elif run and not checkpoint:
    st.warning("Please add checkpoint files to the ckp/ folder and select one.")
elif run and not has_image:
    st.warning("Please upload an image or select an example.")

st.sidebar.divider()
render_system_status()
st.sidebar.caption("<br>If you find this software useful, please cite:<br>" + CITATION, unsafe_allow_html=True)
