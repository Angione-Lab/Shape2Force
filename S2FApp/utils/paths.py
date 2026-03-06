"""Path resolution utilities for S2F App."""
import os


def get_ckp_base(root):
    """Resolve checkpoint base directory (S2FApp/ckp or project/ckp)."""
    ckp_base = os.path.join(root, "ckp")
    if not os.path.isdir(ckp_base):
        project_root = os.path.dirname(root)
        if os.path.isdir(os.path.join(project_root, "ckp")):
            ckp_base = os.path.join(project_root, "ckp")
    return ckp_base


def model_subfolder(model_type):
    """Return subfolder name for model type: 'single_cell' or 'spheroid'."""
    return "single_cell" if model_type == "single_cell" else "spheroid"
