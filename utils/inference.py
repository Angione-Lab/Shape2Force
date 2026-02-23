"""
Inference helpers: run single-cell model on batches, plot samples, save all predictions with metrics.
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from utils import config
from utils.metrics import detect_tanh_output_model, convert_tanh_to_sigmoid_range


def run_batch_singlecell(bf_batch, generator, device, substrate, norm_params, config_path=None):
    """Run single-cell generator on a batch of 1-channel BF images; returns CPU predictions [0,1]."""
    from models.s2f_model import create_settings_channels
    meta = {"substrate": [substrate] * bf_batch.shape[0]}
    settings_ch = create_settings_channels(meta, norm_params, device, bf_batch.shape, config_path=config_path)
    inp = torch.cat([bf_batch.to(device, dtype=torch.float32), settings_ch], dim=1)
    with torch.no_grad():
        pred = generator(inp)
        if detect_tanh_output_model(generator):
            pred = convert_tanh_to_sigmoid_range(pred)
    return pred.cpu()


def force_sum_and_stats(heatmap_np):
    """Return (force_scaled, pixel_sum, max, mean) for a prediction heatmap."""
    pixel_sum = float(np.sum(heatmap_np))
    force_scaled = pixel_sum * config.SCALE_FACTOR_FORCE
    return force_scaled, pixel_sum, float(np.max(heatmap_np)), float(np.mean(heatmap_np))


def plot_inference_samples(loader, generator, n_samples=3, device=None, substrate='fibroblasts_PDMS',
                           normalization_params=None, config_path=None):
    """Plot BF | Prediction for first n_samples from loader (no ground truth)."""
    from utils.substrate_settings import compute_settings_normalization
    device = device or next(generator.parameters()).device
    generator.eval()
    norm = normalization_params or compute_settings_normalization(config_path=config_path)
    bf_list = []
    it = iter(loader)
    while len(bf_list) < n_samples:
        try:
            batch = next(it)
        except StopIteration:
            break
        for i in range(batch.shape[0]):
            if len(bf_list) >= n_samples:
                break
            bf_list.append(batch[i])
    n = len(bf_list)
    if n == 0:
        print("No samples in loader.")
        return
    bf_batch = torch.stack(bf_list)
    pred = run_batch_singlecell(bf_batch, generator, device, substrate, norm, config_path)
    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    for i in range(n):
        axes[i, 0].imshow(bf_list[i].squeeze().numpy(), cmap='gray')
        axes[i, 0].set_title('Bright field')
        axes[i, 0].axis('off')
        axes[i, 1].imshow(pred[i].squeeze().numpy(), cmap='jet', vmin=0, vmax=1)
        axes[i, 1].set_title('Prediction')
        axes[i, 1].axis('off')
    plt.tight_layout()
    plt.show()


def save_all_predictions(loader, generator, save_dir, device=None, substrate='fibroblasts_PDMS',
                         normalization_params=None, config_path=None):
    """Run inference on full loader; save each image as BF | Prediction with force/pixel stats."""
    from utils.substrate_settings import compute_settings_normalization
    device = device or next(generator.parameters()).device
    generator.eval()
    norm = normalization_params or compute_settings_normalization(config_path=config_path)
    paths = loader.dataset.paths
    os.makedirs(save_dir, exist_ok=True)
    idx = 0
    with torch.no_grad():
        for batch in loader:
            pred = run_batch_singlecell(batch, generator, device, substrate, norm, config_path)
            for i in range(pred.shape[0]):
                bf_np = batch[i].squeeze().numpy()
                pred_np = pred[i].squeeze().numpy()
                force_scaled, pixel_sum, hm_max, hm_mean = force_sum_and_stats(pred_np)
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(bf_np, cmap='gray')
                axes[0].set_title('Bright field')
                axes[0].axis('off')
                axes[1].imshow(pred_np, cmap='jet', vmin=0, vmax=1)
                axes[1].set_title('Prediction')
                axes[1].axis('off')
                stats = (f'Force (scaled): {force_scaled:.4f}  |  Pixel sum: {pixel_sum:.2f}  |  '
                         f'Max: {hm_max:.4f}  |  Mean: {hm_mean:.4f}')
                fig.suptitle(stats, fontsize=10)
                plt.tight_layout()
                name = os.path.splitext(os.path.basename(paths[idx]))[0] + '_pred.png'
                plt.savefig(os.path.join(save_dir, name), dpi=150, bbox_inches='tight')
                plt.close()
                idx += 1
    print(f"Saved {idx} predictions to {save_dir}")


def run_batch_spheroid(bf_batch, generator, device):
    """Run spheroid generator on a batch of 1-channel BF images (no settings); returns CPU predictions [0,1]."""
    inp = bf_batch.to(device, dtype=torch.float32)
    with torch.no_grad():
        pred = generator(inp)
        if detect_tanh_output_model(generator):
            pred = convert_tanh_to_sigmoid_range(pred)
    return pred.cpu()


def plot_inference_samples_spheroid(loader, generator, n_samples=3, device=None):
    """Plot BF | Prediction for first n_samples from loader (spheroid, no ground truth)."""
    device = device or next(generator.parameters()).device
    generator.eval()
    bf_list = []
    it = iter(loader)
    while len(bf_list) < n_samples:
        try:
            batch = next(it)
        except StopIteration:
            break
        for i in range(batch.shape[0]):
            if len(bf_list) >= n_samples:
                break
            bf_list.append(batch[i])
    n = len(bf_list)
    if n == 0:
        print("No samples in loader.")
        return
    bf_batch = torch.stack(bf_list)
    pred = run_batch_spheroid(bf_batch, generator, device)
    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    for i in range(n):
        axes[i, 0].imshow(bf_list[i].squeeze().numpy(), cmap='gray')
        axes[i, 0].set_title('Bright field')
        axes[i, 0].axis('off')
        axes[i, 1].imshow(pred[i].squeeze().numpy(), cmap='jet', vmin=0, vmax=1)
        axes[i, 1].set_title('Prediction')
        axes[i, 1].axis('off')
    plt.tight_layout()
    plt.show()


def save_all_predictions_spheroid(loader, generator, save_dir, device=None):
    """Run spheroid inference on full loader; save each image as BF | Prediction with force/pixel stats."""
    device = device or next(generator.parameters()).device
    generator.eval()
    paths = loader.dataset.paths
    os.makedirs(save_dir, exist_ok=True)
    idx = 0
    with torch.no_grad():
        for batch in loader:
            pred = run_batch_spheroid(batch, generator, device)
            for i in range(pred.shape[0]):
                bf_np = batch[i].squeeze().numpy()
                pred_np = pred[i].squeeze().numpy()
                force_scaled, pixel_sum, hm_max, hm_mean = force_sum_and_stats(pred_np)
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(bf_np, cmap='gray')
                axes[0].set_title('Bright field')
                axes[0].axis('off')
                axes[1].imshow(pred_np, cmap='jet', vmin=0, vmax=1)
                axes[1].set_title('Prediction')
                axes[1].axis('off')
                stats = (f'Force (scaled): {force_scaled:.4f}  |  Pixel sum: {pixel_sum:.2f}  |  '
                         f'Max: {hm_max:.4f}  |  Mean: {hm_mean:.4f}')
                fig.suptitle(stats, fontsize=10)
                plt.tight_layout()
                name = os.path.splitext(os.path.basename(paths[idx]))[0] + '_pred.png'
                plt.savefig(os.path.join(save_dir, name), dpi=150, bbox_inches='tight')
                plt.close()
                idx += 1
    print(f"Saved {idx} predictions to {save_dir}")
