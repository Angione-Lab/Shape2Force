"""Result display: single and batch prediction views."""
import csv
import io
import os
import zipfile

import cv2
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.display import apply_display_scale, cv_colormap_to_plotly_colorscale, is_display_range_remapped
from utils.report import heatmap_to_rgb_with_contour, heatmap_to_png_bytes, create_pdf_report
from utils.segmentation import estimate_cell_mask
from ui.heatmaps import render_horizontal_colorbar, add_cell_contour_to_fig
from ui.measure_tool import (
    build_original_vals,
    build_cell_vals,
    render_region_canvas,
    _compute_cell_metrics,
    HAS_DRAWABLE_CANVAS,
)

# Histogram bar color (matches static/s2f_styles.css accent)
_HISTOGRAM_ACCENT = "#0d9488"
_RESULT_FIG_HEIGHT = 320
_HISTOGRAM_HEIGHT = 180


def _result_banner(badge: str, badge_class: str, title: str) -> str:
    """HTML row for INPUT/OUTPUT section titles (batch + single views). badge_class: input | output."""
    return (
        f'<div class="result-label"><span class="result-badge {badge_class}">{badge}</span> {title}</div>'
    )


def render_batch_results(batch_results, colormap_name="Jet", display_mode="Default",
                        clip_min=0, clip_max=1,
                        auto_cell_boundary=False, clamp_only=False):
    """
    Render batch prediction results: summary table, bright-field row, heatmap row, and bulk download.
    batch_results: list of dicts with img, heatmap, force, pixel_sum, key_img, cell_mask.
    cell_mask is computed on-the-fly when auto_cell_boundary is True and not stored.
    """
    if not batch_results:
        return

    # Resolve cell_mask and precompute display_heatmap for each result
    for r in batch_results:
        if auto_cell_boundary and (r.get("cell_mask") is None or not np.any(r.get("cell_mask", 0) > 0)):
            r["_cell_mask"] = estimate_cell_mask(r["heatmap"])
        else:
            r["_cell_mask"] = r.get("cell_mask") if auto_cell_boundary else None
        r["_display_heatmap"] = apply_display_scale(
            r["heatmap"], display_mode,
            clip_min=clip_min, clip_max=clip_max, clamp_only=clamp_only,
        )
    # Build table rows - consistent column names for both modes
    headers = ["Image", "Force", "Sum", "Max", "Mean"]
    rows = []
    csv_rows = [["image"] + headers[1:]]
    for r in batch_results:
        heatmap = r["heatmap"]
        cell_mask = r.get("_cell_mask")
        key = r["key_img"] or "image"
        if auto_cell_boundary and cell_mask is not None and np.any(cell_mask > 0):
            vals = heatmap[cell_mask > 0]
            cell_pixel_sum = float(np.sum(vals))
            cell_force = cell_pixel_sum * (r["force"] / r["pixel_sum"]) if r["pixel_sum"] > 0 else cell_pixel_sum
            cell_mean = cell_pixel_sum / np.sum(cell_mask) if np.sum(cell_mask) > 0 else 0
            row = [key, f"{cell_force:.2f}", f"{cell_pixel_sum:.2f}",
                   f"{np.max(heatmap):.4f}", f"{cell_mean:.4f}"]
        else:
            row = [key, f"{r['force']:.2f}", f"{r['pixel_sum']:.2f}",
                   f"{np.max(heatmap):.4f}", f"{np.mean(heatmap):.4f}"]
        rows.append(row)
        csv_rows.append([os.path.splitext(key)[0]] + row[1:])
    st.markdown(_result_banner("INPUT", "input", "Bright-field images"), unsafe_allow_html=True)
    n_cols = min(5, len(batch_results))
    bf_cols = st.columns(n_cols)
    for i, r in enumerate(batch_results):
        img = r["img"]
        if img.ndim == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with bf_cols[i % n_cols]:
            st.image(img_rgb, caption=r["key_img"], use_container_width=True)
    is_rescale_b = is_display_range_remapped(display_mode, clip_min, clip_max)
    st.markdown(_result_banner("OUTPUT", "output", "Predicted force maps"), unsafe_allow_html=True)
    hm_cols = st.columns(n_cols)
    for i, r in enumerate(batch_results):
        hm_rgb = heatmap_to_rgb_with_contour(
            r["_display_heatmap"], colormap_name,
            r.get("_cell_mask") if auto_cell_boundary else None,
        )
        with hm_cols[i % n_cols]:
            st.image(hm_rgb, caption=r["key_img"], use_container_width=True)
    render_horizontal_colorbar(colormap_name, clip_min, clip_max, is_rescale_b)
    # Table
    st.dataframe(
        {h: [r[i] for r in rows] for i, h in enumerate(headers)},
        use_container_width=True,
        hide_index=True,
    )
    # Histograms in accordion (one per row for visibility)
    with st.expander("Force distribution (histograms)", expanded=False):
        for i, r in enumerate(batch_results):
            heatmap = r["heatmap"]
            cell_mask = r.get("_cell_mask")
            vals = heatmap[cell_mask > 0] if (cell_mask is not None and np.any(cell_mask > 0) and auto_cell_boundary) else heatmap.flatten()
            vals = vals[vals > 0] if np.any(vals > 0) else vals
            st.markdown(f"**{r['key_img']}**")
            if len(vals) > 0:
                fig = go.Figure(data=[go.Histogram(x=vals, nbinsx=50, marker_color=_HISTOGRAM_ACCENT)])
                fig.update_layout(
                    height=_HISTOGRAM_HEIGHT, margin=dict(l=40, r=20, t=10, b=40),
                    xaxis_title="Force value", yaxis_title="Count",
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.caption("No data")
            if i < len(batch_results) - 1:
                st.divider()
    # Bulk downloads: CSV and heatmaps (zip)
    buf_csv = io.StringIO()
    csv.writer(buf_csv).writerows(csv_rows)
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in batch_results:
            hm_bytes = heatmap_to_png_bytes(
                r["_display_heatmap"], colormap_name,
                r.get("_cell_mask") if auto_cell_boundary else None,
            )
            base = os.path.splitext(r["key_img"] or "image")[0]
            zf.writestr(f"{base}_heatmap.png", hm_bytes.getvalue())
    zip_buf.seek(0)
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            "Download all as CSV",
            data=buf_csv.getvalue(),
            file_name="s2f_batch_results.csv",
            mime="text/csv",
            key="download_batch_csv",
            icon=":material/download:",
        )
    with dl_col2:
        st.download_button(
            "Download all heatmaps",
            data=zip_buf.getvalue(),
            file_name="s2f_batch_heatmaps.zip",
            mime="application/zip",
            key="download_batch_heatmaps",
            icon=":material/image:",
        )


def render_result_display(img, raw_heatmap, display_heatmap, pixel_sum, force, key_img, download_key_suffix="",
                         colormap_name="Jet", display_mode="Default", measure_region_dialog=None, auto_cell_boundary=True,
                         cell_mask=None, clip_min=0.0, clip_max=1.0, clamp_only=False):
    """
    Render prediction result: plot, metrics, expander, and download/measure buttons.
    measure_region_dialog: callable to open measure dialog (when ST_DIALOG available).
    auto_cell_boundary: when True, use estimated cell area for metrics; when False, use entire map.
    cell_mask: optional precomputed cell mask; if None and auto_cell_boundary, will be computed.
    """
    if cell_mask is None and auto_cell_boundary:
        cell_mask = estimate_cell_mask(raw_heatmap)
    elif not auto_cell_boundary:
        cell_mask = None
    cell_pixel_sum, cell_force, cell_mean = _compute_cell_metrics(raw_heatmap, cell_mask, pixel_sum, force) if cell_mask is not None else (None, None, None)
    use_cell_metrics = auto_cell_boundary and cell_pixel_sum is not None and cell_force is not None and cell_mean is not None

    base_name = os.path.splitext(key_img or "image")[0]
    if use_cell_metrics:
        main_csv_rows = [
            ["image", "Cell sum", "Cell force (scaled)", "Heatmap max", "Heatmap mean"],
            [base_name, f"{cell_pixel_sum:.2f}", f"{cell_force:.2f}",
             f"{np.max(raw_heatmap):.4f}", f"{cell_mean:.4f}"],
        ]
    else:
        main_csv_rows = [
            ["image", "Sum of all pixels", "Force (scaled)", "Heatmap max", "Heatmap mean"],
            [base_name, f"{pixel_sum:.2f}", f"{force:.2f}",
             f"{np.max(raw_heatmap):.4f}", f"{np.mean(raw_heatmap):.4f}"],
        ]
    buf_main_csv = io.StringIO()
    csv.writer(buf_main_csv).writerows(main_csv_rows)

    buf_hm = heatmap_to_png_bytes(display_heatmap, colormap_name, cell_mask=cell_mask)

    is_rescale = is_display_range_remapped(display_mode, clip_min, clip_max)

    tit1, tit2 = st.columns(2)
    with tit1:
        st.markdown(_result_banner("INPUT", "input", "Bright-field image"), unsafe_allow_html=True)
    with tit2:
        st.markdown(_result_banner("OUTPUT", "output", "Predicted force map"), unsafe_allow_html=True)
    fig_pl = make_subplots(rows=1, cols=2)
    fig_pl.add_trace(go.Heatmap(z=img, colorscale="gray", showscale=False), row=1, col=1)
    plotly_colorscale = cv_colormap_to_plotly_colorscale(colormap_name)
    colorbar_cfg = dict(len=0.4, thickness=12, tickmode="array")
    tick_positions = [0, 0.25, 0.5, 0.75, 1]
    if is_rescale:
        rng = clip_max - clip_min
        colorbar_cfg["tickvals"] = tick_positions
        colorbar_cfg["ticktext"] = [f"{clip_min + t * rng:.2f}" for t in tick_positions]
    else:
        colorbar_cfg["tickvals"] = tick_positions
        colorbar_cfg["ticktext"] = [f"{t:.2f}" for t in tick_positions]
    fig_pl.add_trace(go.Heatmap(z=display_heatmap, colorscale=plotly_colorscale, zmin=0.0, zmax=1.0, showscale=True,
        colorbar=colorbar_cfg), row=1, col=2)
    add_cell_contour_to_fig(fig_pl, cell_mask, row=1, col=2)
    fig_pl.update_layout(
        height=_RESULT_FIG_HEIGHT,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(scaleanchor="y", scaleratio=1),
        xaxis2=dict(scaleanchor="y2", scaleratio=1),
    )
    fig_pl.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig_pl.update_yaxes(showticklabels=False, autorange="reversed", showgrid=False, zeroline=False)
    st.plotly_chart(fig_pl, use_container_width=True, config={"displayModeBar": True, "responsive": True})

    col1, col2, col3, col4 = st.columns(4)
    if use_cell_metrics:
        with col1:
            st.metric("Cell sum", f"{cell_pixel_sum:.2f}", help="Sum over estimated cell area (background excluded)")
        with col2:
            st.metric("Cell force (scaled)", f"{cell_force:.2f}", help="Total traction force in physical units")
        with col3:
            st.metric("Heatmap max", f"{np.max(raw_heatmap):.4f}", help="Peak force intensity in the map")
        with col4:
            st.metric("Heatmap mean", f"{cell_mean:.4f}", help="Mean force over estimated cell area")
    else:
        with col1:
            st.metric("Sum of all pixels", f"{pixel_sum:.2f}", help="Raw sum of all pixel values in the force map")
        with col2:
            st.metric("Force (scaled)", f"{force:.2f}", help="Total traction force in physical units (full field of view)")
        with col3:
            st.metric("Heatmap max", f"{np.max(raw_heatmap):.4f}", help="Peak force intensity in the map")
        with col4:
            st.metric("Heatmap mean", f"{np.mean(raw_heatmap):.4f}", help="Average force intensity (full FOV)")

    # Statistics panel (mean, std, percentiles, histogram)
    with st.expander("Statistics"):
        vals = raw_heatmap[cell_mask > 0] if (cell_mask is not None and np.any(cell_mask > 0) and use_cell_metrics) else raw_heatmap.flatten()
        if len(vals) > 0:
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            p25, p50, p75, p90 = (
                float(np.percentile(vals, 25)), float(np.percentile(vals, 50)),
                float(np.percentile(vals, 75)), float(np.percentile(vals, 90)),
            )
            with stat_col1:
                st.metric("Mean", f"{float(np.mean(vals)):.4f}")
                st.metric("Std", f"{float(np.std(vals)):.4f}")
            with stat_col2:
                st.metric("P25", f"{p25:.4f}")
                st.metric("P50 (median)", f"{p50:.4f}")
            with stat_col3:
                st.metric("P75", f"{p75:.4f}")
                st.metric("P90", f"{p90:.4f}")
            st.markdown("**Histogram**")
            hist_fig = go.Figure(data=[go.Histogram(x=vals, nbinsx=50, marker_color=_HISTOGRAM_ACCENT)])
            hist_fig.update_layout(
                height=_HISTOGRAM_HEIGHT, margin=dict(l=40, r=20, t=20, b=40),
                xaxis_title="Force value", yaxis_title="Count",
                showlegend=False,
            )
            st.plotly_chart(hist_fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.caption("No nonzero values to compute statistics.")

    with st.expander("How to read the results"):
        if use_cell_metrics:
            st.markdown("""
**Input (left):** Bright-field microscopy image of a cell or spheroid on a substrate.  
This is the raw image you provided—it shows cell shape but not forces.

**Output (right):** Predicted traction force map.  
- **Color** indicates force magnitude: blue = low, red = high  
- **Brighter/warmer colors** = stronger forces exerted by the cell on the substrate  
- **Red border = estimated cell area** (background excluded from metrics)
- Values are normalized to [0, 1] for visualization

**Metrics (auto cell boundary on):**
- **Cell sum:** Sum over estimated cell area (background excluded)
- **Cell force (scaled):** Total traction force in physical units
- **Heatmap max:** Peak force intensity in the map
- **Heatmap mean:** Mean force over the estimated cell area
            """)
        else:
            st.markdown("""
**Input (left):** Bright-field microscopy image of a cell or spheroid on a substrate.  
This is the raw image you provided—it shows cell shape but not forces.

**Output (right):** Predicted traction force map.  
- **Color** indicates force magnitude: blue = low, red = high  
- **Brighter/warmer colors** = stronger forces exerted by the cell on the substrate  
- Values are normalized to [0, 1] for visualization

**Metrics (auto cell boundary off):**
- **Sum of all pixels:** Raw sum over entire map
- **Force (scaled):** Total traction force in physical units (full field of view)
- **Heatmap max/mean:** Peak and average force intensity (full field of view)
            """)

    original_vals = build_original_vals(raw_heatmap, pixel_sum, force)

    pdf_bytes = create_pdf_report(
        img, display_heatmap, raw_heatmap, pixel_sum, force, base_name, colormap_name,
        cell_mask=cell_mask, cell_pixel_sum=cell_pixel_sum, cell_force=cell_force, cell_mean=cell_mean
    )

    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
    with btn_col1:
        if HAS_DRAWABLE_CANVAS and measure_region_dialog is not None:
            if st.button("Measure tool", key="open_measure", icon=":material/straighten:"):
                st.session_state["open_measure_dialog"] = True
                st.rerun()
        elif HAS_DRAWABLE_CANVAS:
            with st.expander("Measure tool"):
                expander_cell_vals = build_cell_vals(raw_heatmap, cell_mask, pixel_sum, force) if (auto_cell_boundary and cell_mask is not None) else None
                expander_cell_mask = cell_mask if auto_cell_boundary else None
                render_region_canvas(
                    display_heatmap,
                    raw_heatmap=raw_heatmap,
                    bf_img=img,
                    original_vals=original_vals,
                    cell_vals=expander_cell_vals,
                    cell_mask=expander_cell_mask,
                    key_suffix="expander",
                    input_filename=key_img,
                    colormap_name=colormap_name,
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
    with btn_col4:
        st.download_button(
            "Download report",
            width="stretch",
            data=pdf_bytes,
            file_name=f"{base_name}_report.pdf",
            mime="application/pdf",
            key=f"download_pdf{download_key_suffix}",
            icon=":material/picture_as_pdf:",
        )
