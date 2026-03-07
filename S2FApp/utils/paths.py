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


def get_ckp_folder(ckp_base, model_type):
    """Return checkpoint folder path for model type."""
    return os.path.join(ckp_base, model_subfolder(model_type))


def get_sample_folder(root, model_type):
    """Return sample folder path for model type (samples/<subfolder>)."""
    return os.path.join(root, "samples", model_subfolder(model_type))


def list_files_in_folder(folder, extensions):
    """
    List files in folder matching extensions. Returns sorted list.
    extensions: str or tuple of suffixes, e.g. '.pth' or ('.tif', '.png'). Matching is case-insensitive.
    """
    if not os.path.isdir(folder):
        return []
    ext_tuple = (extensions,) if isinstance(extensions, str) else extensions

    def matches(fname):
        fname_lower = fname.lower()
        return any(fname_lower.endswith(e.lower()) for e in ext_tuple)

    return sorted(f for f in os.listdir(folder) if matches(f))
