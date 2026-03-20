"""UI components for S2F App. Re-exports from submodules for backward compatibility."""
import streamlit as st

# Resolve st.dialog early to fix ordering bug (used in measure dialog)
ST_DIALOG = getattr(st, "dialog", None) or getattr(st, "experimental_dialog", None)

from ui.result_display import render_batch_results, render_result_display
from ui.measure_tool import (
    build_original_vals,
    build_cell_vals,
    render_region_canvas,
    parse_canvas_shapes_to_masks,
    compute_region_metrics,
    HAS_DRAWABLE_CANVAS,
)

__all__ = [
    "ST_DIALOG",
    "HAS_DRAWABLE_CANVAS",
    "render_batch_results",
    "render_result_display",
    "build_original_vals",
    "build_cell_vals",
    "render_region_canvas",
    "parse_canvas_shapes_to_masks",
    "compute_region_metrics",
]
