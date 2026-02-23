"""
Shape2Force (S2F) - GUI for force map prediction from bright field microscopy images.
"""
import os
import sys
import io
import cv2
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

import numpy as np
import streamlit as st
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure S2F is in path
S2F_ROOT = os.path.dirname(os.path.abspath(__file__))
if S2F_ROOT not in sys.path:
    sys.path.insert(0, S2F_ROOT)

from utils.substrate_settings import list_substrates

st.set_page_config(page_title="Shape2Force (S2F)", page_icon="🔬", layout="centered")
st.markdown("""
    <style>
    section[data-testid="stSidebar"] { width: 380px !important; }
    </style>
    """, unsafe_allow_html=True)
st.title("🔬 Shape2Force (S2F)")
st.caption("Predict force maps from bright field microscopy images")

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

SAMPLE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def get_ckp_files_for_model(model_type):
    """Return list of .pth files in the checkpoint folder for the given model type."""
    folder = ckp_single_cell if model_type == "single_cell" else ckp_spheroid
    if os.path.isdir(folder):
        return sorted([f for f in os.listdir(folder) if f.endswith(".pth")])
    return []


def get_sample_files_for_model(model_type):
    """Return list of sample images in the sample folder for the given model type."""
    folder = sample_single_cell if model_type == "single_cell" else sample_spheroid
    if os.path.isdir(folder):
        return sorted([f for f in os.listdir(folder)
                       if f.lower().endswith(SAMPLE_EXTENSIONS)])
    return []

# Sidebar: model configuration
with st.sidebar:
    st.header("Model configuration")
    model_type = st.radio(
        "Model type",
        ["single_cell", "spheroid"],
        format_func=lambda x: "Single cell" if x == "single_cell" else "Spheroid",
        horizontal=False,
        help="Single cell: substrate-aware force prediction. Spheroid: spheroid force maps.",
    )
    st.caption(f"Inference mode: **{'Single cell' if model_type == 'single_cell' else 'Spheroid'}**")

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
            else:
                substrate_config = None
        except FileNotFoundError:
            st.error("config/substrate_settings.json not found")

    st.divider()
    st.subheader("Display")
    display_size = st.slider("Image size (px)", min_value=200, max_value=800, value=350, step=50,
                             help="Adjust display size. Drag to pan, scroll to zoom.")

    st.divider()

# Main area: image input
img_source = st.radio("Image source", ["Upload", "Sample"], horizontal=True, label_visibility="collapsed")
img = None
uploaded = None
selected_sample = None

if img_source == "Upload":
    uploaded = st.file_uploader(
        "Upload bright field image",
        type=["tif", "tiff", "png", "jpg", "jpeg"],
        help="Bright field microscopy image (grayscale or RGB)",
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
            "Select sample image",
            sample_files,
            format_func=lambda x: x,
            key=f"sample_{model_type}",
        )
        if selected_sample:
            sample_path = os.path.join(sample_folder, selected_sample)
            img = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
        # Show sample thumbnails (filtered by model type)
        st.caption(f"Sample images from `samples/{sample_subfolder_name}/`")
        n_cols = min(5, len(sample_files))
        cols = st.columns(n_cols)
        for i, fname in enumerate(sample_files[:8]):  # show up to 8
            with cols[i % n_cols]:
                path = os.path.join(sample_folder, fname)
                sample_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if sample_img is not None:
                    st.image(sample_img, caption=fname, width='content')
    else:
        st.info(f"No sample images in samples/{sample_subfolder_name}/. Add images or use Upload.")

run = st.button("Run prediction", type="primary")
has_image = img is not None

if run and checkpoint and has_image:
    st.markdown(f"**Using checkpoint:** `ckp/{ckp_subfolder_name}/{checkpoint}`")
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

                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Sum of all pixels", f"{pixel_sum:.2f}")
                with col2:
                    st.metric("Cell force (scaled)", f"{force:.2f}")
                with col3:
                    st.metric("Heatmap max", f"{np.max(heatmap):.4f}")
                with col4:
                    st.metric("Heatmap mean", f"{np.mean(heatmap):.4f}")

                # Visualization - Plotly with zoom/pan
                fig_pl = make_subplots(rows=1, cols=2, subplot_titles=["", ""])
                fig_pl.add_trace(go.Heatmap(z=img, colorscale="gray", showscale=False), row=1, col=1)
                fig_pl.add_trace(go.Heatmap(z=heatmap, colorscale="Jet", zmin=0, zmax=1, showscale=True), row=1, col=2)
                fig_pl.update_layout(
                    height=display_size,
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(scaleanchor="y", scaleratio=1),
                    xaxis2=dict(scaleanchor="y2", scaleratio=1),
                )
                fig_pl.update_xaxes(showticklabels=False)
                fig_pl.update_yaxes(showticklabels=False, autorange="reversed")
                st.plotly_chart(fig_pl, use_container_width=True)

                # Download
                heatmap_uint8 = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
                heatmap_rgb = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB)
                pil_heatmap = Image.fromarray(heatmap_rgb)
                buf_hm = io.BytesIO()
                pil_heatmap.save(buf_hm, format="PNG")
                buf_hm.seek(0)
                st.download_button("Download Heatmap", data=buf_hm.getvalue(),
                                   file_name="s2f_heatmap.png", mime="image/png")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            import traceback
            st.code(traceback.format_exc())

elif run and not checkpoint:
    st.warning("Please add checkpoint files to the ckp/ folder and select one.")
elif run and not has_image:
    st.warning("Please upload an image or select a sample.")

# Footer
st.sidebar.divider()
st.sidebar.caption("Checkpoints: ckp/single_cell/ and ckp/spheroid/. Samples: samples/single_cell/ and samples/spheroid/")
