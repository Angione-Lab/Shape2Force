"""
Shape2Force (S2F) - GUI for force map prediction from bright field microscopy images.
"""
import os
import sys
import traceback

# Suppress OpenCV verbose logging (cv2.utils.logging not reliably available in all builds)
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2

import numpy as np
import streamlit as st

S2F_ROOT = os.path.dirname(os.path.abspath(__file__))
if S2F_ROOT not in sys.path:
    sys.path.insert(0, S2F_ROOT)

from config.constants import (
    BATCH_INFERENCE_SIZE,
    BATCH_MAX_IMAGES,
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
    render_batch_results,
    render_result_display,
    render_region_canvas,
    ST_DIALOG,
    HAS_DRAWABLE_CANVAS,
)

CITATION = (
    "Lautaro Baro, Kaveh Shahhosseini, Amparo Andrés-Bordería, Claudio Angione, and Maria Angeles Juanes. "
    "<b>\"Shape-to-force (S2F): Predicting Cell Traction Forces from LabelFree Imaging\"</b>, 2026."
)


def _inference_cache_condition_key(model_type, use_manual, substrate_val, substrate_config):
    """Hashable key for substrate / manual conditions so cache invalidates when single-cell inputs change."""
    if model_type != "single_cell":
        return None
    if use_manual and substrate_config is not None:
        return (
            "manual",
            round(float(substrate_config["pixelsize"]), 6),
            round(float(substrate_config["young"]), 2),
        )
    return ("preset", str(substrate_val))


# Measure tool dialog: defined early so it exists before render_result_display uses it
if HAS_DRAWABLE_CANVAS and ST_DIALOG:
    @ST_DIALOG("Measure tool", width="medium")
    def measure_region_dialog():
        raw_heatmap = st.session_state.get("measure_raw_heatmap")
        if raw_heatmap is None:
            st.warning("No prediction available to measure.")
            return
        display_mode = st.session_state.get("measure_display_mode", "Default")
        _m_clamp = st.session_state.get(
            "measure_clamp_only", st.session_state.get("measure_clip_bounds", False)
        )
        display_heatmap = apply_display_scale(
            raw_heatmap, display_mode,
            min_percentile=st.session_state.get("measure_min_percentile", 0),
            max_percentile=st.session_state.get("measure_max_percentile", 100),
            clip_min=st.session_state.get("measure_clip_min", 0),
            clip_max=st.session_state.get("measure_clip_max", 1),
            clamp_only=_m_clamp,
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
                                    min_percentile=0, max_percentile=100, clip_min=0, clip_max=1,
                                    clamp_only=False):
    """Populate session state for the measure tool. If cell_mask is None and auto_cell_boundary, computes it."""
    if cell_mask is None and auto_cell_boundary:
        cell_mask = estimate_cell_mask(heatmap)
    st.session_state["measure_raw_heatmap"] = heatmap.copy()
    st.session_state["measure_display_mode"] = display_mode
    st.session_state["measure_min_percentile"] = min_percentile
    st.session_state["measure_max_percentile"] = max_percentile
    st.session_state["measure_clip_min"] = clip_min
    st.session_state["measure_clip_max"] = clip_max
    st.session_state["measure_clamp_only"] = clamp_only
    st.session_state["measure_bf_img"] = img.copy()
    st.session_state["measure_input_filename"] = key_img or "image"
    st.session_state["measure_original_vals"] = build_original_vals(heatmap, pixel_sum, force)
    st.session_state["measure_colormap"] = colormap_name
    st.session_state["measure_auto_cell_on"] = auto_cell_boundary
    st.session_state["measure_cell_vals"] = build_cell_vals(heatmap, cell_mask, pixel_sum, force) if auto_cell_boundary else None
    st.session_state["measure_cell_mask"] = cell_mask if auto_cell_boundary else None


st.set_page_config(page_title="Shape2Force (S2F)", page_icon="🦠", layout="wide")

st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)

_css_path = os.path.join(S2F_ROOT, "static", "s2f_styles.css")
if os.path.exists(_css_path):
    with open(_css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
<div class="s2f-header">
    <h1>🦠 Shape2Force (S2F)</h1>
    <p>Predict traction force maps from bright-field microscopy images of cells or spheroids</p>
</div>
""", unsafe_allow_html=True)

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


def _render_sample_selector(model_type, batch_mode):
    """
    Render sample image selector (Example mode). Returns (img, imgs_batch, selected_sample, selected_samples).
    For single mode: img is set, imgs_batch=[]. For batch: img=None, imgs_batch=list of (img, key).
    """
    sample_folder = get_sample_folder(S2F_ROOT, model_type)
    sample_files = list_files_in_folder(sample_folder, SAMPLE_EXTENSIONS)
    sample_subfolder_name = model_subfolder(model_type)
    img = None
    imgs_batch = []
    selected_sample = None
    selected_samples = []

    if not sample_files:
        st.info(f"No example images in samples/{sample_subfolder_name}/. Add images or use Upload.")
        return img, imgs_batch, selected_sample, selected_samples

    if batch_mode:
        selected_samples = st.multiselect(
            f"Select example images (max {BATCH_MAX_IMAGES})",
            sample_files,
            default=None,
            max_selections=BATCH_MAX_IMAGES,
            key=f"sample_batch_{model_type}",
        )
        if selected_samples:
            for fname in selected_samples[:BATCH_MAX_IMAGES]:
                sample_path = os.path.join(sample_folder, fname)
                loaded = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
                if loaded is not None:
                    imgs_batch.append((loaded, fname))
    else:
        selected_sample = st.selectbox(
            f"Select example image (from `samples/{sample_subfolder_name}/`)",
            sample_files,
            format_func=lambda x: x,
            key=f"sample_{model_type}",
        )
        if selected_sample:
            sample_path = os.path.join(sample_folder, selected_sample)
            img = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)

    thumbnails = get_cached_sample_thumbnails(model_type, sample_folder, sample_files)
    n_cols = min(5, len(thumbnails))
    cols = st.columns(n_cols)
    for i, (fname, sample_img) in enumerate(thumbnails):
        if sample_img is not None:
            with cols[i % n_cols]:
                st.image(sample_img, caption=fname, width=120)
    return img, imgs_batch, selected_sample, selected_samples


# Sidebar
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <span class="brand-text">Shape2Force</span>
    </div>
    """, unsafe_allow_html=True)

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
            key=f"checkpoint_{model_type}",
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

    batch_mode = st.toggle(
        "Batch mode",
        value=False,
        help=f"Process up to {BATCH_MAX_IMAGES} images at once. Upload multiple files or select multiple examples.",
    )

    auto_cell_boundary = st.toggle(
        "Auto boundary",
        value=False,
        help="When on: estimate cell region from force map and use it for metrics (red contour). When off: use entire map.",
    )

    force_scale_mode = st.radio(
        "Force scale",
        ["Default", "Range"],
        horizontal=True,
        key="s2f_force_scale",
        help="Default: display forces on the full 0–1 scale. Range: set a sub-range; values outside are zeroed and the rest is stretched to the colormap.",
    )
    if force_scale_mode == "Default":
        clip_min, clip_max = 0.0, 1.0
        display_mode = "Default"
        clamp_only = True
    else:
        mn_col, mx_col = st.columns(2)
        with mn_col:
            clip_min = st.number_input(
                "Min",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.01,
                format="%.2f",
                key="s2f_clip_min",
                help="Lower bound of the display range (0–1).",
            )
        with mx_col:
            clip_max = st.number_input(
                "Max",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.01,
                format="%.2f",
                key="s2f_clip_max",
                help="Upper bound of the display range (0–1).",
            )
        if clip_min >= clip_max:
            st.warning("Min must be less than max. Using 0.00–1.00 for display.")
            clip_min, clip_max = 0.0, 1.0
        display_mode = "Range"
        clamp_only = False
    min_percentile, max_percentile = 0, 100

    cm_col_lbl, cm_col_sb = st.columns([1, 2])
    with cm_col_lbl:
        st.markdown('<p class="selectbox-label">Colormap</p>', unsafe_allow_html=True)
    with cm_col_sb:
        colormap_name = st.selectbox(
            "Colormap",
            list(COLORMAPS.keys()),
            key="s2f_colormap",
            label_visibility="collapsed",
            help="Color scheme for the force map. Viridis is often preferred for accessibility.",
        )


# Main area: image input
img_source = st.radio("Image source", ["Upload", "Example"], horizontal=True, label_visibility="collapsed", key="s2f_img_source")
img = None
imgs_batch = []  # list of (img, key_img) for batch mode
uploaded = None
uploaded_list = []
selected_sample = None
selected_samples = []

if batch_mode:
    # Batch mode: multiple images (max BATCH_MAX_IMAGES)
    if img_source == "Upload":
        uploaded_list = st.file_uploader(
            "Upload bright-field images",
            type=["tif", "tiff", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            help=f"Select up to {BATCH_MAX_IMAGES} images. Bright-field microscopy (grayscale or RGB).",
        )
        if uploaded_list:
            uploaded_list = uploaded_list[:BATCH_MAX_IMAGES]
            for u in uploaded_list:
                bytes_data = u.read()
                nparr = np.frombuffer(bytes_data, np.uint8)
                decoded = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                if decoded is not None:
                    imgs_batch.append((decoded, u.name))
                u.seek(0)
    else:
        img, imgs_batch, selected_sample, selected_samples = _render_sample_selector(model_type, batch_mode=True)
else:
    # Single image mode
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
        img, imgs_batch, selected_sample, selected_samples = _render_sample_selector(model_type, batch_mode=False)

st.markdown("")
col_btn, col_info = st.columns([1, 3])
with col_btn:
    run = st.button("Run prediction", type="primary", use_container_width=True)
with col_info:
    ckp_path = f"ckp/{ckp_subfolder_name}/{checkpoint}" if checkpoint else f"ckp/{ckp_subfolder_name}/"
    st.markdown(f"""
    <div class="run-info">
        <span class="run-info-tag">{MODEL_TYPE_LABELS[model_type]}</span>
        <code>{ckp_path}</code>
    </div>
    """, unsafe_allow_html=True)

has_image = img is not None
has_batch = len(imgs_batch) > 0

if "prediction_result" not in st.session_state:
    st.session_state["prediction_result"] = None
if "batch_results" not in st.session_state:
    st.session_state["batch_results"] = None
if not batch_mode:
    st.session_state["batch_results"] = None  # Clear when switching to single mode

# Single-image keys (for non-batch)
key_img = (uploaded.name if uploaded else None) if img_source == "Upload" else selected_sample
_cond_key = _inference_cache_condition_key(model_type, use_manual, substrate_val, substrate_config)
current_key = (model_type, checkpoint, key_img, _cond_key)
cached = st.session_state["prediction_result"]
has_cached = cached is not None and cached.get("cache_key") == current_key and not batch_mode
just_ran = run and checkpoint and has_image and not batch_mode
just_ran_batch = run and checkpoint and has_batch and batch_mode


@st.cache_resource
def _load_predictor(model_type, checkpoint, ckp_folder):
    """Load and cache predictor. Invalidated when model_type or checkpoint changes."""
    from predictor import S2FPredictor
    return S2FPredictor(
        model_type=model_type,
        checkpoint_path=checkpoint,
        ckp_folder=ckp_folder,
    )


def _prepare_and_render_cached_result(r, key_img, colormap_name, display_mode, auto_cell_boundary,
                                     min_percentile, max_percentile, clip_min, clip_max, clamp_only,
                                     download_key_suffix="", check_measure_dialog=False,
                                     show_success=False):
    """Prepare display from cached result and render. Used by both just_ran and has_cached paths."""
    img, heatmap, force, pixel_sum = r["img"], r["heatmap"], r["force"], r["pixel_sum"]
    display_heatmap = apply_display_scale(
        heatmap, display_mode,
        min_percentile=min_percentile,
        max_percentile=max_percentile,
        clip_min=clip_min,
        clip_max=clip_max,
        clamp_only=clamp_only,
    )
    cell_mask = estimate_cell_mask(heatmap) if auto_cell_boundary else None
    _populate_measure_session_state(
        heatmap, img, pixel_sum, force, key_img, colormap_name,
        display_mode, auto_cell_boundary, cell_mask=cell_mask,
        min_percentile=min_percentile, max_percentile=max_percentile,
        clip_min=clip_min, clip_max=clip_max, clamp_only=clamp_only,
    )
    if check_measure_dialog and st.session_state.pop("open_measure_dialog", False):
        measure_region_dialog()
    if show_success:
        st.success("Prediction complete!")
    render_result_display(
        img, heatmap, display_heatmap, pixel_sum, force, key_img,
        download_key_suffix=download_key_suffix,
        colormap_name=colormap_name,
        display_mode=display_mode,
        measure_region_dialog=_get_measure_dialog_fn(),
        auto_cell_boundary=auto_cell_boundary,
        cell_mask=cell_mask,
        clip_min=clip_min, clip_max=clip_max, clamp_only=clamp_only,
    )


if just_ran_batch:
    st.session_state["prediction_result"] = None
    st.session_state["batch_results"] = None
    with st.spinner("Loading model and predicting..."):
        progress_bar = None
        try:
            predictor = _load_predictor(model_type, checkpoint, ckp_folder)
            sub_val = substrate_val if model_type == "single_cell" and not use_manual else DEFAULT_SUBSTRATE
            n_images = len(imgs_batch)
            progress_bar = st.progress(0, text=f"Predicting 0 / {n_images} images")
            pred_results = []
            for start in range(0, n_images, BATCH_INFERENCE_SIZE):
                chunk = imgs_batch[start : start + BATCH_INFERENCE_SIZE]
                chunk_results = predictor.predict_batch(
                    chunk,
                    substrate=sub_val,
                    substrate_config=substrate_config if model_type == "single_cell" else None,
                )
                pred_results.extend(chunk_results)
                progress_bar.progress(min(start + len(chunk), n_images) / n_images,
                                     text=f"Predicting {len(pred_results)} / {n_images} images")
            batch_results = [
                {
                    "img": img_b.copy(),
                    "heatmap": heatmap.copy(),
                    "force": force,
                    "pixel_sum": pixel_sum,
                    "key_img": key_b,
                    "cell_mask": estimate_cell_mask(heatmap) if auto_cell_boundary else None,
                }
                for (img_b, key_b), (heatmap, force, pixel_sum) in zip(imgs_batch, pred_results)
            ]
            st.session_state["batch_results"] = batch_results
            progress_bar.empty()
            st.success(f"Prediction complete for {len(batch_results)} image(s)!")
            render_batch_results(
                batch_results,
                colormap_name=colormap_name,
                display_mode=display_mode,
                min_percentile=min_percentile,
                max_percentile=max_percentile,
                clip_min=clip_min,
                clip_max=clip_max,
                auto_cell_boundary=auto_cell_boundary,
                clamp_only=clamp_only,
            )
        except Exception as e:
            if progress_bar is not None:
                progress_bar.empty()
            st.error(f"Prediction failed: {e}")
            st.code(traceback.format_exc())

elif batch_mode and st.session_state.get("batch_results"):
    render_batch_results(
        st.session_state["batch_results"],
        colormap_name=colormap_name,
        display_mode=display_mode,
        min_percentile=min_percentile,
        max_percentile=max_percentile,
        clip_min=clip_min,
        clip_max=clip_max,
        auto_cell_boundary=auto_cell_boundary,
        clamp_only=clamp_only,
    )

elif just_ran:
    st.session_state["prediction_result"] = None
    with st.spinner("Loading model and predicting..."):
        try:
            predictor = _load_predictor(model_type, checkpoint, ckp_folder)
            sub_val = substrate_val if model_type == "single_cell" and not use_manual else DEFAULT_SUBSTRATE
            heatmap, force, pixel_sum = predictor.predict(
                image_array=img,
                substrate=sub_val,
                substrate_config=substrate_config if model_type == "single_cell" else None,
            )
            cache_key = (model_type, checkpoint, key_img, _cond_key)
            r = {
                "img": img.copy(),
                "heatmap": heatmap.copy(),
                "force": force,
                "pixel_sum": pixel_sum,
                "cache_key": cache_key,
            }
            st.session_state["prediction_result"] = r
            _prepare_and_render_cached_result(
                r, key_img, colormap_name, display_mode, auto_cell_boundary,
                min_percentile, max_percentile, clip_min, clip_max, clamp_only,
                download_key_suffix="", check_measure_dialog=False,
                show_success=True,
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.code(traceback.format_exc())

elif has_cached:
    r = st.session_state["prediction_result"]
    _prepare_and_render_cached_result(
        r, key_img, colormap_name, display_mode, auto_cell_boundary,
        min_percentile, max_percentile, clip_min, clip_max, clamp_only,
        download_key_suffix="_cached", check_measure_dialog=True,
        show_success=False,
    )

elif run and not checkpoint:
    st.warning("Please add checkpoint files to the ckp/ folder and select one.")
elif run and not has_image and not has_batch:
    st.warning("Please upload an image or select an example.")
elif run and batch_mode and not has_batch:
    st.warning(f"Please upload or select 1–{BATCH_MAX_IMAGES} images for batch processing.")

st.markdown(f"""
<div class="footer-citation">
    <span>If you find this software useful, please cite: {CITATION}</span>
</div>
""", unsafe_allow_html=True)
