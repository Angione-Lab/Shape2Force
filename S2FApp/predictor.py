"""
Core inference logic for S2F (Shape2Force).
Predicts force maps from bright field microscopy images.
"""
import hashlib
import os
import sys
import cv2
import torch
import numpy as np

# Ensure S2F is in path when running from project root or S2F
S2F_ROOT = os.path.dirname(os.path.abspath(__file__))
if S2F_ROOT not in sys.path:
    sys.path.insert(0, S2F_ROOT)

from config.constants import BATCH_INFERENCE_SIZE, DEFAULT_SUBSTRATE, MODEL_INPUT_SIZE
from models.s2f_model import create_s2f_model
from utils.paths import get_ckp_base, model_subfolder
from utils.substrate_settings import get_settings_of_category, compute_settings_normalization
from utils import config


def load_image(filepath, target_size=None):
    """Load and preprocess a bright field image."""
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {filepath}")
    size = target_size if target_size is not None else MODEL_INPUT_SIZE
    if isinstance(size, int):
        size = (size, size)
    img = cv2.resize(img, size)
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


def file_sha256(path, chunk=1 << 20):
    """SHA-256 of a file (recorded in reports so a prediction can be traced to its checkpoint)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def detect_architecture(state_dict):
    """
    Generator architecture from the checkpoint keys:
      's2f'          S2FGenerator (multi-scale input 'scale_pyramid.*'); tanh output -> (y + 1) / 2
      's2f_spheroid' supervised attention U-Net ('encoder1.*'); sigmoid output in [0, 1]
    """
    keys = list(state_dict.keys())
    if any(k.startswith("scale_pyramid.") or k.startswith("initial_conv.") for k in keys):
        return "s2f"
    if any(k.startswith("encoder1.") for k in keys):
        return "s2f_spheroid"
    raise ValueError("Unrecognised checkpoint: neither S2FGenerator nor the supervised spheroid model")


class S2FPredictor:
    """
    Shape2Force predictor for single-cell or spheroid force map prediction.
    The generator architecture is selected from the checkpoint itself, so both spheroid checkpoints
    (ckp_spheroid_GN.pth = S2F generator as in the paper, ckp_spheroid_FN.pth = smaller supervised model) load.
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
        ckp_base = get_ckp_base(S2F_ROOT)
        subfolder = model_subfolder(model_type)
        ckp_dir = ckp_folder if ckp_folder else os.path.join(ckp_base, subfolder)
        if not os.path.isdir(ckp_dir):
            ckp_dir = ckp_base  # fallback if subfolders not used
        if not checkpoint_path:
            raise ValueError("checkpoint_path is required")

        full_path = checkpoint_path
        if not os.path.isabs(checkpoint_path):
            full_path = os.path.join(ckp_dir, checkpoint_path)
        if not os.path.exists(full_path):
            full_path = os.path.join(ckp_base, checkpoint_path)  # try base folder
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Checkpoint not found: {full_path}")

        checkpoint = torch.load(full_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("generator_state_dict") or checkpoint.get("model_state_dict") or checkpoint
        self.architecture = detect_architecture(state)
        self.in_channels = 3 if model_type == "single_cell" else 1
        generator, _ = create_s2f_model(in_channels=self.in_channels, model_type=self.architecture)
        if model_type == "single_cell":
            generator.load_checkpoint_with_expansion(full_path, strict=True)
        else:
            generator.load_state_dict(state, strict=True)
        if self.architecture == "s2f_spheroid" and hasattr(generator, "set_output_mode"):
            generator.set_output_mode(use_tanh=False)  # trained with sigmoid output in [0, 1]
        self._use_tanh_output = self.architecture == "s2f"  # S2FGenerator: tanh in [-1, 1]

        self.generator = generator.to(self.device).eval()
        self.checkpoint_name = os.path.basename(full_path)
        self.checkpoint_sha256 = file_sha256(full_path)
        self.checkpoint_epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
        self.n_params = sum(p.numel() for p in self.generator.parameters())
        self.norm_params = compute_settings_normalization() if model_type == "single_cell" else None
        self.config_path = os.path.join(S2F_ROOT, "config", "substrate_settings.json")

    def describe(self):
        """Model information for reports."""
        return {
            "model_type": self.model_type,
            "architecture": self.architecture,
            "checkpoint": self.checkpoint_name,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_epoch": self.checkpoint_epoch,
            "n_params_M": round(self.n_params / 1e6, 1),
            "device": str(self.device),
        }

    def substrate_channels(self, substrate=None, substrate_config=None):
        """Values of the two substrate input channels (normalised pixel-size and Young's-modulus settings)."""
        if self.model_type != "single_cell":
            return None
        ch = create_settings_channels_single(substrate if substrate is not None else DEFAULT_SUBSTRATE, "cpu", 1, 1,
                                             config_path=self.config_path, substrate_config=substrate_config)
        return {"pixelsize_norm": float(ch[0, 0, 0, 0]), "young_norm": float(ch[0, 1, 0, 0])}

    def predict(self, image_path=None, image_array=None, substrate=None,
                substrate_config=None):
        """
        Run prediction on an image.

        Args:
            image_path: Path to bright field image (tif, png, jpg)
            image_array: numpy array (H, W) or (H, W, C) in [0, 255] or [0, 1]
            substrate: Substrate name for single-cell mode (used if substrate_config is None)
            substrate_config: Optional dict with 'pixelsize' and 'young'. Overrides substrate lookup.

        Returns:
            heatmap: numpy array (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE) in [0, 1]
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
            img = cv2.resize(img, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
        else:
            raise ValueError("Provide image_path or image_array")

        x = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(self.device)  # [1,1,H,W]

        if self.model_type == "single_cell" and self.norm_params is not None:
            sub = substrate if substrate is not None else DEFAULT_SUBSTRATE
            settings_ch = create_settings_channels_single(
                sub, self.device, x.shape[2], x.shape[3],
                config_path=self.config_path, substrate_config=substrate_config
            )
            x = torch.cat([x, settings_ch], dim=1)  # [1,3,H,W]

        with torch.no_grad():
            pred = self.generator(x)

        if self._use_tanh_output:
            pred = (pred + 1.0) / 2.0  # Tanh [-1,1] to [0, 1]
        # else: the supervised spheroid model already outputs sigmoid [0, 1]
        heatmap = pred[0, 0].cpu().numpy()
        force = sum_force_map(pred).item()
        pixel_sum = float(np.sum(heatmap))

        return heatmap, force, pixel_sum

    def predict_batch(self, images, substrate=None, substrate_config=None, batch_size=None,
                     on_progress=None):
        """
        Run prediction on a batch of images. Processes in chunks to avoid OOM on
        memory-constrained environments (e.g. Hugging Face free tier).

        Args:
            images: List of (img_array, key) or list of img arrays. img_array: (H, W) or (H, W, C).
            substrate: Substrate name for single-cell mode (same for all images).
            substrate_config: Optional dict with 'pixelsize' and 'young' (same for all).
            batch_size: Max images per forward pass (default: BATCH_INFERENCE_SIZE). Use 1 for minimal memory.
            on_progress: Optional callback(processed: int, total: int) called after each forward pass.

        Returns:
            List of (heatmap, force, pixel_sum) tuples.
        """
        batch_size = batch_size if batch_size is not None else BATCH_INFERENCE_SIZE
        imgs = []
        for item in images:
            img = item[0] if isinstance(item, tuple) else item
            img = np.asarray(img, dtype=np.float32)
            if img.ndim == 3:
                img = img[:, :, 0] if img.shape[-1] >= 1 else img
            if img.max() > 1.0:
                img = img / 255.0
            img = cv2.resize(img, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
            imgs.append(img)

        results = []
        for start in range(0, len(imgs), batch_size):
            chunk = imgs[start : start + batch_size]
            x = torch.from_numpy(np.stack(chunk)).float().unsqueeze(1).to(self.device)  # [B, 1, H, W]

            if self.model_type == "single_cell" and self.norm_params is not None:
                sub = substrate if substrate is not None else DEFAULT_SUBSTRATE
                settings_ch = create_settings_channels_single(
                    sub, self.device, x.shape[2], x.shape[3],
                    config_path=self.config_path, substrate_config=substrate_config
                )
                settings_batch = settings_ch.expand(x.shape[0], -1, -1, -1)
                x = torch.cat([x, settings_batch], dim=1)  # [B, 3, H, W]

            with torch.no_grad():
                pred = self.generator(x)

            if self._use_tanh_output:
                pred = (pred + 1.0) / 2.0

            for i in range(pred.shape[0]):
                heatmap = pred[i, 0].cpu().numpy()
                force = sum_force_map(pred[i : i + 1]).item()
                pixel_sum = float(np.sum(heatmap))
                results.append((heatmap, force, pixel_sum))

            if on_progress is not None:
                on_progress(len(results), len(imgs))

        return results
