"""
Substrate settings for force map prediction.
Loads from config/substrate_settings.json - users can edit this file to add/modify substrates.
"""
import os
import json


def _default_config_path():
    """Default path to substrate settings config (S2F/config/substrate_settings.json)."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(this_dir)  # S2F root
    return os.path.join(project_root, 'config', 'substrate_settings.json')


def load_substrate_config(config_path=None):
    """
    Load substrate settings from config file.

    Args:
        config_path: Path to JSON config. If None, uses config/substrate_settings.json in S2F root.

    Returns:
        dict: Config with 'substrates', 'default_substrate'
    """
    path = config_path or _default_config_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Substrate config not found at {path}. "
            "Create config/substrate_settings.json or pass config_path."
        )
    with open(path, 'r') as f:
        return json.load(f)


def resolve_substrate(name, config=None, config_path=None):
    """
    Resolve substrate name to a canonical substrate key.

    Args:
        name: Substrate key (e.g. 'fibroblasts_PDMS', 'PDMS_10kPa')
        config: Pre-loaded config dict. If None, loads from config_path.
        config_path: Path to config file (used if config is None).

    Returns:
        str: Canonical substrate key
    """
    if config is None:
        config = load_substrate_config(config_path)

    s = (name or '').strip()
    if not s:
        return config.get('default_substrate', 'Fibroblasts_Fibronectin_6KPa')

    substrates = config.get('substrates', {})
    s_lower = s.lower()
    for key in substrates:
        if key.lower() == s_lower:
            return key
    for key in substrates:
        if s_lower.startswith(key.lower()) or key.lower().startswith(s_lower):
            return key

    return config.get('default_substrate', 'Fibroblasts_Fibronectin_6KPa')


def get_settings_of_category(substrate_name, config=None, config_path=None):
    """
    Get pixelsize and young's modulus for a substrate.

    Args:
        substrate_name: Substrate or folder name (case-insensitive)
        config: Pre-loaded config dict. If None, loads from config_path.
        config_path: Path to config file (used if config is None).

    Returns:
        dict: {'name': str, 'pixelsize': float, 'young': float}
    """
    if config is None:
        config = load_substrate_config(config_path)

    substrate_key = resolve_substrate(substrate_name, config=config)
    substrates = config.get('substrates', {})
    default = config.get('default_substrate', 'Fibroblasts_Fibronectin_6KPa')

    if substrate_key in substrates:
        return substrates[substrate_key].copy()

    default_settings = substrates.get(default, {'name': 'Fibroblasts on Fibronectin (6 kPa)', 'pixelsize': 3.0769, 'young': 6000})
    return default_settings.copy()


def list_substrates(config=None, config_path=None):
    """
    Return list of available substrate keys for user selection.

    Returns:
        list: Substrate keys
    """
    if config is None:
        config = load_substrate_config(config_path)
    return list(config.get('substrates', {}).keys())


def compute_settings_normalization(config=None, config_path=None):
    """
    Compute min-max normalization parameters from all substrates in config.

    Returns:
        dict: {'pixelsize': {'min', 'max'}, 'young': {'min', 'max'}}
    """
    if config is None:
        config = load_substrate_config(config_path)

    substrates = config.get('substrates', {})
    all_pixelsizes = [s['pixelsize'] for s in substrates.values()]
    all_youngs = [s['young'] for s in substrates.values()]

    if not all_pixelsizes or not all_youngs:
        pixelsize_min, pixelsize_max = 3.0769, 9.8138
        young_min, young_max = 1000.0, 10000.0
    else:
        pixelsize_min, pixelsize_max = min(all_pixelsizes), max(all_pixelsizes)
        young_min, young_max = min(all_youngs), max(all_youngs)

    return {
        'pixelsize': {'min': pixelsize_min, 'max': pixelsize_max},
        'young': {'min': young_min, 'max': young_max}
    }
