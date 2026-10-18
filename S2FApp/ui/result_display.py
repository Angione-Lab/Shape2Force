"""Result display: single and batch prediction views (traction stress in Pa, areas in µm², forces in nN)."""
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
from utils.report import (create_batch_pdf_report, create_pdf_report, heatmap_figure_png_bytes,
                          heatmap_to_rgb_with_contour, info_rows)
from utils.segmentation import estimate_cell_mask
from utils.units import fmt, make_scale, metric_rows, nice_scale_bar_um, region_metrics, to_pa
from ui.heatmaps import colorbar_ticks_pa, render_horizontal_colorbar, add_cell_contour_to_fig
from ui.measure_tool import (
    build_original_vals,
    build_cell_vals,
    render_region_canvas,
    HAS_DRAWABLE_CANVAS,
)

# Histogram bar color (matches static/s2f_styles.css accent)
_HISTOGRAM_ACCENT = "#0d9488"
_RESULT_FIG_HEIGHT = 340
_HISTOGRAM_HEIGHT = 180
_METRIC_HELP = {
    "Mean traction stress": "Mean predicted traction-stress magnitude over the region (Pa)",
    "Max traction stress": "Peak predicted traction-stress magnitude (Pa)",
    "Area": "Region area in µm² (needs the image pixel size)",
    "Total force": "Sum of traction stress × pixel area, in nN (needs the image pixel size). Magnitudes only: "
                   "vector cancellation is not taken into account",
}


def _result_banner(badge: str, badge_class: str, title: str) -> str:
    """HTML row for INPUT/OUTPUT section titles (batch + single views). badge_class: input | output."""
    return (
        f'<div class="result-label"><span class="result-badge {badge_class}">{badge}</span> {title}</div>'
    )


def _metrics_csv(base_name, metrics_full, metrics_cell, info):
    """CSV: one row of metrics with units, then the parameters needed to reproduce the prediction."""
    rows = [["image", "region", "mean_stress_Pa", "max_stress_Pa", "area_um2", "total_force_nN", "n_pixels"]]
    for region, m in [("whole_field", metrics_full), ("force_region", metrics_cell)]:
        if m:
            rows.append([base_name, region, fmt(m["mean_pa"], nd=2), fmt(m["max_pa"], nd=2), fmt(m["area_um2"], nd=1),
                         fmt(m["total_force_nN"], nd=3), m["n_px"]])
    rows += [[], ["parameter", "value"]] + [[k.strip(), v] for k, v in info_rows(info)]
    buf = io.StringIO()
    csv.writer(buf).writerows(rows)
    return buf.getvalue()


def _add_scale_bar_to_fig(fig, width_px, height_px, um_per_px, row, col):
    """Draw a µm scale bar (bottom right) on a Plotly image subplot whose axes are in pixels."""
    if not um_per_px:
        return
    length_um = nice_scale_bar_um(width_px * um_per_px)
    if not length_um:
        return
    length_px = length_um / um_per_px
    m = 0.05 * width_px
    x1, y = width_px - m, height_px - m
    xref = "x" if (row, col) == (1, 1) else "x2"
    yref = "y" if (row, col) == (1, 1) else "y2"
    fig.add_shape(type="line", x0=x1 - length_px, x1=x1, y0=y, y1=y, xref=xref, yref=yref,
                  line=dict(color="white", width=4))
    fig.add_annotation(x=x1 - length_px / 2, y=y - 0.03 * height_px, xref=xref, yref=yref, text=f"{length_um:g} µm",
                       showarrow=False, font=dict(color="white", size=11))


def render_batch_results(batch_results, colormap_name="Jet", display_mode="Default",
                         clip_min=0, clip_max=1,
                         auto_cell_boundary=False, clamp_only=False):
    """
    Render batch prediction results: summary table (units), bright-field row, heatmap row and bulk downloads.
    batch_results: list of dicts with img, heatmap, key_img, cell_mask, scale, info.
    """
    if not batch_results:
        return

    for r in batch_results:
        if auto_cell_boundary and (r.get("cell_mask") is None or not np.any(r.get("cell_mask", 0) > 0)):
            r["_cell_mask"] = estimate_cell_mask(r["heatmap"])
        else:
            r["_cell_mask"] = r.get("cell_mask") if auto_cell_boundary else None
        r["_display_heatmap"] = apply_display_scale(r["heatmap"], display_mode, clip_min=clip_min, clip_max=clip_max)
        r["_metrics"] = region_metrics(r["heatmap"], r["_cell_mask"] if auto_cell_boundary else None, r["scale"])

    region = "force region (red contour)" if auto_cell_boundary else "whole field"
    headers = ["Image", "Mean stress (Pa)", "Max stress (Pa)", "Area (µm²)", "Total force (nN)"]
    rows = []
    csv_rows = [["image", "region", "mean_stress_Pa", "max_stress_Pa", "area_um2", "total_force_nN", "n_pixels",
                 "model", "checkpoint", "checkpoint_sha256", "substrate_setting", "image_um_per_px", "stress_max_Pa",
                 "app_version"]]
    for r in batch_results:
        m, key, info = r["_metrics"], r["key_img"] or "image", r.get("info", {})
        rows.append([key, fmt(m["mean_pa"], nd=1), fmt(m["max_pa"], nd=1), fmt(m["area_um2"], nd=0),
                     fmt(m["total_force_nN"], nd=2)])
        sub = info.get("substrate") or {}
        csv_rows.append([os.path.splitext(key)[0], region, fmt(m["mean_pa"], nd=2), fmt(m["max_pa"], nd=2),
                         fmt(m["area_um2"], nd=1), fmt(m["total_force_nN"], nd=3), m["n_px"],
                         info.get("model", {}).get("architecture", ""), info.get("model", {}).get("checkpoint", ""),
                         info.get("model", {}).get("checkpoint_sha256", ""),
                         sub.get("name") or (f"manual {sub.get('pixelsize')}/{sub.get('young')}" if sub else ""),
                         info.get("image_um_per_px") or "", info.get("stress_max_pa", ""), info.get("app_version", "")])
    st.markdown(_result_banner("INPUT", "input", "Bright-field images"), unsafe_allow_html=True)
    n_cols = min(5, len(batch_results))
    bf_cols = st.columns(n_cols)
    for i, r in enumerate(batch_results):
        img = r["img"]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with bf_cols[i % n_cols]:
            st.image(img_rgb, caption=r["key_img"], use_container_width=True)
    is_rescale_b = is_display_range_remapped(display_mode, clip_min, clip_max)
    st.markdown(_result_banner("OUTPUT", "output", "Predicted traction-stress maps"), unsafe_allow_html=True)
    hm_cols = st.columns(n_cols)
    for i, r in enumerate(batch_results):
        hm_rgb = heatmap_to_rgb_with_contour(r["_display_heatmap"], colormap_name,
                                             r.get("_cell_mask") if auto_cell_boundary else None)
        with hm_cols[i % n_cols]:
            st.image(hm_rgb, caption=r["key_img"], use_container_width=True)
    smax = batch_results[0]["scale"]["stress_max_pa"]
    render_horizontal_colorbar(colormap_name, clip_min, clip_max, is_rescale_b, stress_max_pa=smax)
    st.dataframe({h: [row[i] for row in rows] for i, h in enumerate(headers)}, use_container_width=True, hide_index=True)
    st.caption(f"Metrics over the {region}. Area and total force need the image pixel size (sidebar).")
    with st.expander("Traction-stress distribution (histograms)", expanded=False):
        for i, r in enumerate(batch_results):
            cell_mask = r.get("_cell_mask")
            vals = to_pa(r["heatmap"], r["scale"])
            vals = vals[cell_mask > 0] if (cell_mask is not None and np.any(cell_mask > 0) and auto_cell_boundary) else vals.ravel()
            vals = vals[vals > 0] if np.any(vals > 0) else vals
            st.markdown(f"**{r['key_img']}**")
            if len(vals) > 0:
                fig = go.Figure(data=[go.Histogram(x=vals, nbinsx=50, marker_color=_HISTOGRAM_ACCENT)])
                fig.update_layout(height=_HISTOGRAM_HEIGHT, margin=dict(l=40, r=20, t=10, b=40),
                                  xaxis_title="Traction stress (Pa)", yaxis_title="Pixels", showlegend=False)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.caption("No data")
            if i < len(batch_results) - 1:
                st.divider()
    buf_csv = io.StringIO()
    csv.writer(buf_csv).writerows(csv_rows)
    lo_b, hi_b = (clip_min, clip_max) if is_rescale_b else (0.0, 1.0)
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in batch_results:
            base = os.path.splitext(r["key_img"] or "image")[0]
            png = heatmap_figure_png_bytes(r["_display_heatmap"], colormap_name,
                                           r.get("_cell_mask") if auto_cell_boundary else None, r["scale"], lo_b, hi_b)
            zf.writestr(f"{base}_traction_stress.png", png.getvalue())
            zf.writestr(f"{base}_traction_stress_Pa.tif", _tiff_pa_bytes(r["heatmap"], r["scale"]))
        zf.writestr("README.txt", _ZIP_README.format(smax=smax))
    zip_buf.seek(0)
    pdf_items = [dict(img=r["img"], display_heatmap=r["_display_heatmap"], raw_heatmap=r["heatmap"],
                      base_name=os.path.splitext(r["key_img"] or "image")[0],
                      cell_mask=r.get("_cell_mask") if auto_cell_boundary else None,
                      metrics_full=region_metrics(r["heatmap"], None, r["scale"]),
                      metrics_cell=r["_metrics"] if auto_cell_boundary else None,
                      info=r.get("info"), scale=r["scale"]) for r in batch_results]
    pdf_bytes = create_batch_pdf_report(pdf_items, colormap_name, lo_b, hi_b, auto_cell_boundary)
    dl_col1, dl_col2, dl_col3 = st.columns(3)
    with dl_col1:
        st.download_button("Download report (PDF)", data=pdf_bytes, file_name="s2f_batch_report.pdf",
                           mime="application/pdf", key="download_batch_pdf", icon=":material/picture_as_pdf:",
                           help="Summary table of all images, then one page per image with maps, metrics and parameters.")
    with dl_col2:
        st.download_button("Download values (CSV)", data=buf_csv.getvalue(), file_name="s2f_batch_results.csv",
                           mime="text/csv", key="download_batch_csv", icon=":material/download:",
                           help="One row per image: metrics with units and the parameters used.")
    with dl_col3:
        st.download_button("Download maps (ZIP)", data=zip_buf.getvalue(), file_name="s2f_batch_maps.zip",
                           mime="application/zip", key="download_batch_heatmaps", icon=":material/image:",
                           help="For each image: a PNG to view (colourbar and scale bar) and a TIFF whose pixel "
                                "values are traction stress in Pa, for measurements in Fiji/ImageJ.")


_ZIP_README = """Shape2Force (S2F) - predicted traction-stress maps

For each image:
  <name>_traction_stress.png     picture of the map with colourbar (Pa) and scale bar, for viewing and figures
  <name>_traction_stress_Pa.tif  the same map as numbers: each pixel value is the predicted traction stress in Pa
                                 (32-bit, 1024 x 1024). Open it in Fiji/ImageJ to measure regions yourself
                                 (e.g. draw an ROI and use Analyze > Measure: Mean = mean traction stress in Pa).

Scale: model output 1.0 = {smax:.0f} Pa. S2F predicts the magnitude of the traction stress, not its direction.
The TIFF covers the whole image resized to 1024 x 1024 pixels; one TIFF pixel = (image width x pixel size / 1024) um.
"""


def _tiff_pa_bytes(heatmap, scale):
    """Traction-stress map in Pa as a 32-bit float TIFF (readable by Fiji/ImageJ)."""
    ok, enc = cv2.imencode(".tif", to_pa(heatmap, scale).astype(np.float32))
    return enc.tobytes() if ok else b""


def render_result_display(img, raw_heatmap, display_heatmap, pixel_sum, force, key_img, download_key_suffix="",
                          colormap_name="Jet", display_mode="Default", measure_region_dialog=None, auto_cell_boundary=True,
                          cell_mask=None, clip_min=0.0, clip_max=1.0, clamp_only=False, scale=None, info=None):
    """
    Render one prediction: bright field + traction-stress map (µm scale bars, colourbar in Pa), metrics with
    units, statistics, and downloads (PNG with scale bar and colourbar, CSV and PDF report with all parameters).
    scale: utils.units.make_scale(...); info: provenance dict (see app.py).
    """
    scale = scale or make_scale("single_cell", image_shape=img.shape[:2])
    smax = scale["stress_max_pa"]
    if cell_mask is None and auto_cell_boundary:
        cell_mask = estimate_cell_mask(raw_heatmap)
    elif not auto_cell_boundary:
        cell_mask = None
    metrics_full = region_metrics(raw_heatmap, None, scale)
    metrics_cell = region_metrics(raw_heatmap, cell_mask, scale) if (cell_mask is not None and np.any(cell_mask > 0)) else None
    shown = metrics_cell if (auto_cell_boundary and metrics_cell) else metrics_full
    base_name = os.path.splitext(key_img or "image")[0]
    is_rescale = is_display_range_remapped(display_mode, clip_min, clip_max)
    lo, hi = (clip_min, clip_max) if is_rescale else (0.0, 1.0)

    tit1, tit2 = st.columns(2)
    with tit1:
        st.markdown(_result_banner("INPUT", "input", "Bright-field image"), unsafe_allow_html=True)
    with tit2:
        st.markdown(_result_banner("OUTPUT", "output", "Predicted traction-stress map"), unsafe_allow_html=True)
    fig_pl = make_subplots(rows=1, cols=2)
    fig_pl.add_trace(go.Heatmap(z=img, colorscale="gray", showscale=False, hoverinfo="skip"), row=1, col=1)
    ticks, labels = colorbar_ticks_pa(lo, hi, smax)
    colorbar_cfg = dict(len=0.8, thickness=12, tickmode="array", tickvals=ticks, ticktext=labels,
                        title=dict(text="Pa", side="top"))
    fig_pl.add_trace(go.Heatmap(z=display_heatmap, colorscale=cv_colormap_to_plotly_colorscale(colormap_name),
                                zmin=0.0, zmax=1.0, showscale=True, colorbar=colorbar_cfg,
                                customdata=to_pa(raw_heatmap, scale),
                                hovertemplate="x %{x}, y %{y}<br>%{customdata:.1f} Pa<extra></extra>"), row=1, col=2)
    add_cell_contour_to_fig(fig_pl, cell_mask, row=1, col=2)
    _add_scale_bar_to_fig(fig_pl, img.shape[1], img.shape[0], scale.get("image_um_per_px"), 1, 1)
    _add_scale_bar_to_fig(fig_pl, display_heatmap.shape[1], display_heatmap.shape[0], scale.get("hm_um_per_px_x"), 1, 2)
    fig_pl.update_layout(height=_RESULT_FIG_HEIGHT, margin=dict(l=10, r=10, t=10, b=10),
                         xaxis=dict(scaleanchor="y", scaleratio=1), xaxis2=dict(scaleanchor="y2", scaleratio=1))
    fig_pl.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig_pl.update_yaxes(showticklabels=False, autorange="reversed", showgrid=False, zeroline=False)
    st.plotly_chart(fig_pl, use_container_width=True, config={"displayModeBar": True, "responsive": True})
    if is_rescale:
        st.caption(f"Display range {lo * smax:.0f}–{hi * smax:.0f} Pa; stresses above {hi * smax:.0f} Pa are shown "
                   "in the top colour. Metrics always use the full prediction.")

    cols = st.columns(4)
    for col, (label, value) in zip(cols, metric_rows(shown)):
        with col:
            st.metric(label, value, help=_METRIC_HELP[label])
    st.caption(("Metrics inside the force region (red contour), estimated from the predicted map." if shown is metrics_cell else
                "Metrics over the whole field of view.")
               + ("" if scale.get("hm_px_area_um2") else " Enter the image pixel size in the sidebar to get area and total force."))

    with st.expander("Statistics"):
        vals = to_pa(raw_heatmap, scale)
        vals = vals[cell_mask > 0] if (shown is metrics_cell and cell_mask is not None) else vals.ravel()
        if len(vals) > 0:
            c1, c2, c3 = st.columns(3)
            p25, p50, p75, p90 = (float(np.percentile(vals, q)) for q in (25, 50, 75, 90))
            with c1:
                st.metric("Mean", f"{float(np.mean(vals)):.1f} Pa")
                st.metric("Std", f"{float(np.std(vals)):.1f} Pa")
            with c2:
                st.metric("P25", f"{p25:.1f} Pa")
                st.metric("P50 (median)", f"{p50:.1f} Pa")
            with c3:
                st.metric("P75", f"{p75:.1f} Pa")
                st.metric("P90", f"{p90:.1f} Pa")
            hist_fig = go.Figure(data=[go.Histogram(x=vals, nbinsx=50, marker_color=_HISTOGRAM_ACCENT)])
            hist_fig.update_layout(height=_HISTOGRAM_HEIGHT, margin=dict(l=40, r=20, t=20, b=40),
                                   xaxis_title="Traction stress (Pa)", yaxis_title="Pixels", showlegend=False)
            st.plotly_chart(hist_fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.caption("No values to compute statistics.")

    with st.expander("Parameters of this prediction"):
        st.table({"Parameter": [k for k, _ in info_rows(info)], "Value": [str(v) for _, v in info_rows(info)]})

    with st.expander("How to read the results"):
        st.markdown(f"""
**Input (left):** the bright-field image you provided (cell shape only, no force information).

**Output (right):** predicted **traction-stress magnitude** (Pa).
- The model output is normalised to 0–1, with 1 = **{smax:.0f} Pa**, the fixed colour limit of the
  {'single-cell' if smax >= 500 else 'spheroid'} ground-truth TFM maps used for training. The colourbar shows Pa.
- S2F predicts the magnitude only, not the direction, and predictions tend to underestimate the highest stresses.
- **Range** (sidebar) narrows the displayed window, e.g. 0–250 Pa, to show low-stress patterns; stresses above the
  window saturate. The metrics are always computed from the full prediction.
- **Red contour** ("Measure force region only", on by default): the region of high predicted stress, estimated from
  the map (not the cell outline); the metrics refer to this region so the background does not dilute them.

**Metrics:** mean and maximum traction stress (Pa); area (µm²) and total force (nN = Σ stress × pixel area)
need the image pixel size. Total force sums magnitudes, so it is not the net force.
        """)

    original_vals = build_original_vals(raw_heatmap, pixel_sum, force, scale)
    buf_hm = heatmap_figure_png_bytes(display_heatmap, colormap_name, cell_mask, scale, lo, hi)
    csv_text = _metrics_csv(base_name, metrics_full, metrics_cell, info)
    pdf_bytes = create_pdf_report(img, display_heatmap, raw_heatmap, base_name, colormap_name, cell_mask=cell_mask,
                                  metrics_full=metrics_full, metrics_cell=metrics_cell, info=info, scale=scale,
                                  clip_min=lo, clip_max=hi)

    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
    with btn_col1:
        if HAS_DRAWABLE_CANVAS and measure_region_dialog is not None:
            if st.button("Measure tool", key="open_measure", icon=":material/straighten:"):
                st.session_state["open_measure_dialog"] = True
                st.rerun()
        elif HAS_DRAWABLE_CANVAS:
            with st.expander("Measure tool"):
                expander_cell_vals = build_cell_vals(raw_heatmap, cell_mask, pixel_sum, force, scale) if (auto_cell_boundary and cell_mask is not None) else None
                render_region_canvas(display_heatmap, raw_heatmap=raw_heatmap, bf_img=img, original_vals=original_vals,
                                     cell_vals=expander_cell_vals, cell_mask=cell_mask if auto_cell_boundary else None,
                                     key_suffix="expander", input_filename=key_img, colormap_name=colormap_name)
        else:
            st.caption("Install `streamlit-drawable-canvas-fix` for region measurement: `pip install streamlit-drawable-canvas-fix`")
    with btn_col2:
        st.download_button("Download map", width="stretch", data=buf_hm.getvalue(),
                           file_name=f"{base_name}_traction_stress.png", mime="image/png",
                           key=f"download_heatmap{download_key_suffix}", icon=":material/download:")
    with btn_col3:
        st.download_button("Download values", width="stretch", data=csv_text, file_name=f"{base_name}_values.csv",
                           mime="text/csv", key=f"download_main_values{download_key_suffix}", icon=":material/download:")
    with btn_col4:
        st.download_button("Download report", width="stretch", data=pdf_bytes, file_name=f"{base_name}_report.pdf",
                           mime="application/pdf", key=f"download_pdf{download_key_suffix}",
                           icon=":material/picture_as_pdf:")
