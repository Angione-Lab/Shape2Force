"""
Core inference logic for S2F (Shape2Force).
Predicts force maps from bright field microscopy images.
"""
import os
import sys
import cv2
import torch
import numpy as np

# Ensure S2F is in path when running from project root or S2F
S2F_ROOT = os.path.dirname(os.path.abspath(__file__))
if S2F_ROOT not in sys.path:
    sys.path.insert(0, S2F_ROOT)

from models.s2f_model import create_s2f_model
from utils.substrate_settings import get_settings_of_category, compute_settings_normalization
from utils import config


def load_image(filepath, target_size=1024):
    """Load and preprocess a bright field image."""
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {filepath}")
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    return img


def sum_force_map(force_map):
    """Compute cell force as sum of pixel values scaled by SCALE_FACTOR_FORCE."""
    if isinstance(force_map, np.ndarray):
        force_map = torch.from_numpy(force_map.astype(np.float32))
    if force_map.dim() == 2:
        force_map = force_map.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    elif force_map.dim() == 3:
        force_map = force_map.unsqueeze(0)  # [1, 1, H, W]
    # force_map: [B, 1, H, W], sum over spatial dims (2, 3)
    return torch.sum(force_map, dim=(2, 3)) * config.SCALE_FACTOR_FORCE


def create_settings_channels_single(substrate_name, device, height, width, config_path=None,
                                    substrate_config=None):
    """
    Create settings channels for a single image (single-cell mode).

    Args:
        substrate_name: Substrate name (used if substrate_config is None)
        device: torch device
        height, width: spatial dimensions
        config_path: Path to substrate config JSON
        substrate_config: Optional dict with 'pixelsize' and 'young'. If provided, overrides substrate_name.
    """
    norm_params = compute_settings_normalization(config_path=config_path)
    if substrate_config is not None and 'pixelsize' in substrate_config and 'young' in substrate_config:
        settings = substrate_config
    else:
        settings = get_settings_of_category(substrate_name, config_path=config_path)
    pmin, pmax = norm_params['pixelsize']['min'], norm_params['pixelsize']['max']
    ymin, ymax = norm_params['young']['min'], norm_params['young']['max']
    pixelsize_norm = (settings['pixelsize'] - pmin) / (pmax - pmin) if pmax > pmin else 0.5
    young_norm = (settings['young'] - ymin) / (ymax - ymin) if ymax > ymin else 0.5
    pixelsize_norm = max(0.0, min(1.0, pixelsize_norm))
    young_norm = max(0.0, min(1.0, young_norm))
    pixelsize_ch = torch.full(
        (1, 1, height, width), pixelsize_norm, device=device, dtype=torch.float32
    )
    young_ch = torch.full(
        (1, 1, height, width), young_norm, device=device, dtype=torch.float32
    )
    return torch.cat([pixelsize_ch, young_ch], dim=1)


class S2FPredictor:
    """
    Shape2Force predictor for single-cell or spheroid force map prediction.
    """

    def __init__(self, model_type="single_cell", checkpoint_path=None, ckp_folder=None, device=None):
        """
        Args:
            model_type: "single_cell" or "spheroid"
            checkpoint_path: Path to .pth checkpoint (relative to ckp_folder or absolute)
            ckp_folder: Folder containing checkpoints (default: S2F/ckp)
            device: "cuda" or "cpu" (auto-detected if None)
        """
        self.model_type = model_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckp_base = os.path.join(S2F_ROOT, "ckp")
        if not os.path.isdir(ckp_base):
            project_root = os.path.dirname(S2F_ROOT)
            if os.path.isdir(os.path.join(project_root, "ckp")):
                ckp_base = os.path.join(project_root, "ckp")
        subfolder = "single_cell" if model_type == "single_cell" else "spheroid"
        ckp_dir = ckp_folder if ckp_folder else os.path.join(ckp_base, subfolder)
        if not os.path.isdir(ckp_dir):
            ckp_dir = ckp_base  # fallback if subfolders not used

        in_channels = 3 if model_type == "single_cell" else 1
        s2f_model_type = "s2f" if model_type == "single_cell" else "s2f_spheroid"
        generator, _ = create_s2f_model(in_channels=in_channels, model_type=s2f_model_type)
        self.generator = generator

        if checkpoint_path:
            full_path = checkpoint_path
            if not os.path.isabs(checkpoint_path):
                full_path = os.path.join(ckp_dir, checkpoint_path)
            if not os.path.exists(full_path):
                full_path = os.path.join(ckp_base, checkpoint_path)  # try base folder
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"Checkpoint not found: {full_path}")

            if model_type == "single_cell":
                self.generator.load_checkpoint_with_expansion(full_path, strict=True)
            else:
                checkpoint = torch.load(full_path, map_location="cpu", weights_only=False)
                state = checkpoint.get("generator_state_dict") or checkpoint.get("model_state_dict") or checkpoint
                self.generator.load_state_dict(state, strict=True)
                if hasattr(self.generator, "set_output_mode"):
                    self.generator.set_output_mode(use_tanh=False)  # sigmoid [0,1] for inference

        self.generator = self.generator.to(self.device)
        self.generator.eval()

        self.norm_params = compute_settings_normalization() if model_type == "single_cell" else None
        self._use_tanh_output = model_type == "single_cell"  # single_cell uses tanh, spheroid uses sigmoid
        self.config_path = os.path.join(S2F_ROOT, "config", "substrate_settings.json")

    def predict(self, image_path=None, image_array=None, substrate="Fibroblasts_Fibronectin_6KPa",
                substrate_config=None):
        """
        Run prediction on an image.

        Args:
            image_path: Path to bright field image (tif, png, jpg)
            image_array: numpy array (H, W) or (H, W, C) in [0, 255] or [0, 1]
            substrate: Substrate name for single-cell mode (used if substrate_config is None)
            substrate_config: Optional dict with 'pixelsize' and 'young'. Overrides substrate lookup.

        Returns:
            heatmap: numpy array (1024, 1024) in [0, 1]
            force: scalar cell force (sum of heatmap * SCALE_FACTOR_FORCE)
            pixel_sum: raw sum of all pixel values in heatmap
        """
        if image_path is not None:
            img = load_image(image_path)
        elif image_array is not None:
            img = np.asarray(image_array, dtype=np.float32)
            if img.ndim == 3:
                img = img[:, :, 0] if img.shape[-1] >= 1 else img
            if img.max() > 1.0:
                img = img / 255.0
            img = cv2.resize(img, (1024, 1024))
        else:
            raise ValueError("Provide image_path or image_array")

        x = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(self.device)  # [1,1,H,W]

        if self.model_type == "single_cell" and self.norm_params is not None:
            settings_ch = create_settings_channels_single(
                substrate, self.device, x.shape[2], x.shape[3],
                config_path=self.config_path, substrate_config=substrate_config
            )
            x = torch.cat([x, settings_ch], dim=1)  # [1,3,H,W]

        with torch.no_grad():
            pred = self.generator(x)

        if self._use_tanh_output:
            pred = (pred + 1.0) / 2.0  # Tanh [-1,1] to [0, 1]
        # else: spheroid already outputs sigmoid [0, 1]
        heatmap = pred[0, 0].cpu().numpy()
        force = sum_force_map(pred).item()
        pixel_sum = float(np.sum(heatmap))

        return heatmap, force, pixel_sum
