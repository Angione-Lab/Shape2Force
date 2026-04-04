"""Metrics for S2F training and evaluation.

Includes: MSE, MS-SSIM, Pixel Correlation (Pearson), Relative Magnitude Error (WFM),
and evaluation helpers for notebooks and scripts.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr
from tqdm import tqdm
import matplotlib.pyplot as plt

try:
    from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure
    from torchmetrics import MeanSquaredError
    HAS_TORCHMETRICS = True
except ImportError:
    HAS_TORCHMETRICS = False


def calculate_mse(y_true, y_pred):
    if isinstance(y_true, torch.Tensor):
        return F.mse_loss(y_pred, y_true).item()
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def calculate_psnr(y_true, y_pred, max_pixel_value=1.0):
    mse = calculate_mse(y_true, y_pred)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_pixel_value / np.sqrt(mse))


def calculate_ssim_tensor(y_true, y_pred, data_range=1.0):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    ssim_values = []
    batch_size = y_true.shape[0]
    for i in range(batch_size):
        if len(y_true.shape) == 4:
            true_img = y_true[i, 0] if y_true.shape[1] == 1 else y_true[i, 0]
            pred_img = y_pred[i, 0] if y_pred.shape[1] == 1 else y_pred[i, 0]
        else:
            true_img, pred_img = y_true[i], y_pred[i]
        ssim_values.append(ssim(true_img, pred_img, data_range=data_range))
    return np.mean(ssim_values)


def calculate_pearson_correlation(y_true, y_pred):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    correlation, _ = pearsonr(y_true.flatten(), y_pred.flatten())
    return correlation


def calculate_individual_pixel_correlation(y_true, y_pred):
    """Pixel-wise Pearson correlation per sample in batch."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    correlations = []
    batch_size = y_true.shape[0]
    for i in range(batch_size):
        true_flat = y_true[i].flatten()
        pred_flat = y_pred[i].flatten()
        r, _ = pearsonr(true_flat, pred_flat)
        correlations.append(r)
    return correlations


# --- WFM (Wrinkle Force Microscopy) metrics for heatmap as magnitude ---

def _to_numpy_wfm(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _ensure_shape_wfm(f):
    """Ensure (N, 2, H, W). Heatmap -> fx=magnitude, fy=0."""
    if f.ndim == 3:
        if f.shape[-1] == 2:
            f = np.transpose(f, (2, 0, 1))[None, ...]
        elif f.shape[0] == 2:
            f = f[None, ...]
        else:
            raise ValueError(f"Unsupported 3D shape {f.shape}")
    elif f.ndim == 4:
        if f.shape[-1] == 2:
            f = np.transpose(f, (0, 3, 1, 2))
    else:
        raise ValueError(f"Unsupported ndim={f.ndim}")
    return f


def _force_mag_wfm(f):
    fx, fy = f[:, 0], f[:, 1]
    return np.sqrt(fx**2 + fy**2)


def _force_magnitude_tensor(x: torch.Tensor) -> torch.Tensor:
    """Per-pixel force magnitude: (B,1,H,W) uses channel 0 as magnitude; (B,2+,H,W) uses sqrt(fx^2+fy^2)."""
    if x.dim() != 4:
        raise ValueError(f"Expected 4D tensor, got shape {tuple(x.shape)}")
    c = x.size(1)
    if c == 1:
        return x[:, 0]
    if c >= 2:
        return torch.sqrt(x[:, 0].pow(2) + x[:, 1].pow(2))
    raise ValueError(f"Expected at least 1 channel, got {c}")


class WFMRMELoss(nn.Module):
    """Weighted force-magnitude relative error; matches ``wfm_relative_magnitude_error`` (differentiable)."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mag_t = _force_magnitude_tensor(target)
        mag_p = _force_magnitude_tensor(pred)
        if mag_t.shape != mag_p.shape:
            raise ValueError(f"Shape mismatch after magnitude: {mag_t.shape} vs {mag_p.shape}")
        fbar = mag_t.mean().clamp_min(self.eps)
        w = mag_t / fbar
        rel = (mag_p - mag_t).abs() / (mag_t + self.eps)
        return (w * rel).mean()


def wfm_correlation(y_true, y_pred, mode="magnitude"):
    """Pearson correlation between prediction and ground truth (magnitude mode for heatmaps)."""
    t = _ensure_shape_wfm(_to_numpy_wfm(y_true))
    p = _ensure_shape_wfm(_to_numpy_wfm(y_pred))
    if t.shape != p.shape:
        raise ValueError(f"Shape mismatch: true {t.shape} vs pred {p.shape}")
    if mode == "magnitude":
        tv = _force_mag_wfm(t).ravel()
        pv = _force_mag_wfm(p).ravel()
    else:
        raise ValueError(f"Unknown mode '{mode}'")
    tv, pv = tv.astype(np.float64), pv.astype(np.float64)
    if np.allclose(tv.std(), 0) or np.allclose(pv.std(), 0):
        return 0.0
    return float(np.corrcoef(tv, pv)[0, 1])


def wfm_relative_magnitude_error(y_true, y_pred, eps=1e-8):
    """Relative magnitude error for heatmap-as-magnitude."""
    t = _ensure_shape_wfm(_to_numpy_wfm(y_true))
    p = _ensure_shape_wfm(_to_numpy_wfm(y_pred))
    if t.shape != p.shape:
        raise ValueError(f"Shape mismatch: true {t.shape} vs pred {p.shape}")
    mag_t = _force_mag_wfm(t)
    mag_p = _force_mag_wfm(p)
    fbar = np.mean(mag_t)
    if np.isclose(fbar, 0):
        return 0.0
    rel = np.abs(mag_p - mag_t) / (mag_t + eps)
    w = mag_t / fbar
    return float(np.mean(rel * w))


def apply_threshold_mask(tensor, threshold=0.0):
    return tensor * (tensor >= threshold).float()


def detect_tanh_output_model(model):
    """Detect if model outputs [-1, 1] (Tanh)."""
    if hasattr(model, 'use_sigmoid') and not model.use_sigmoid:
        return True
    if hasattr(model, 'use_tanh_output') and model.use_tanh_output:
        return True
    if hasattr(model, 'final_conv'):
        fc = model.final_conv
        if isinstance(fc, nn.Sequential):
            if isinstance(fc[-1], nn.Tanh):
                return True
        elif isinstance(fc, nn.Tanh):
            return True
    return False


def convert_tanh_to_sigmoid_range(tensor):
    return (tensor + 1.0) / 2.0


# --- TorchMetrics wrapper for MS-SSIM ---

class TorchMetricsWrapper:
    def __init__(self, device='cpu'):
        self.device = device
        self.reset_metrics()

    def reset_metrics(self):
        if HAS_TORCHMETRICS:
            self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
            self.mse = MeanSquaredError().to(self.device)
        else:
            self.ms_ssim = None
            self.mse = None

    def compute_ms_ssim(self, y_true, y_pred):
        if not HAS_TORCHMETRICS:
            return float(calculate_ssim_tensor(y_true, y_pred))  # fallback to SSIM
        y_true = y_true.to(self.device)
        y_pred = y_pred.to(self.device)
        if y_true.shape[1] == 1:
            pass
        else:
            y_true, y_pred = y_true[:, 0:1], y_pred[:, 0:1]
        return self.ms_ssim(y_pred, y_true).item()

    def compute_mse(self, y_true, y_pred):
        if not HAS_TORCHMETRICS:
            return calculate_mse(y_true, y_pred)
        y_true = y_true.to(self.device)
        y_pred = y_pred.to(self.device)
        return self.mse(y_pred, y_true).item()


# --- Full evaluation on dataset ---

def evaluate_metrics_on_dataset(generator, data_loader, device=None, description="Evaluating",
                               save_predictions=False, threshold=0.0, use_settings=False,
                               normalization_params=None, config_path=None, substrate_override=None):
    """
    Evaluate S2F generator on a dataset. Returns MSE, MS-SSIM, Pixel Correlation,
    Relative Magnitude Error, and force sum/mean correlations.
    """
    if device is None:
        device = torch.device('mps' if torch.backends.mps.is_available() else
                              'cuda' if torch.cuda.is_available() else 'cpu')

    generator = generator.to(device)
    generator.eval()
    metrics_wrapper = TorchMetricsWrapper(device=device)

    heatmap_mse = []
    heatmap_ms_ssim = []
    heatmap_pixel_corr = []
    wfm_corr_mag = []
    wfm_rel_mag_err = []
    force_sum_gt, force_sum_pred = [], []
    force_mean_gt, force_mean_pred = [], []
    individual_predictions = [] if save_predictions else None

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(data_loader, desc=description)):
            if len(batch_data) == 5:
                images, heatmaps, _, _, metadata = batch_data
                has_metadata = True
            else:
                images, heatmaps, _, _ = batch_data
                has_metadata = False

            images = images.to(device, dtype=torch.float32)
            heatmaps = heatmaps.to(device, dtype=torch.float32)

            if use_settings and normalization_params is not None:
                from models.s2f_model import create_settings_channels
                meta = metadata if has_metadata else {'substrate': [substrate_override or 'fibroblasts_PDMS'] * images.size(0)}
                settings_ch = create_settings_channels(meta, normalization_params, device, images.shape, config_path=config_path)
                images = torch.cat([images, settings_ch], dim=1)

            pred = generator(images)
            if detect_tanh_output_model(generator):
                pred = convert_tanh_to_sigmoid_range(pred)

            gt_thresh = apply_threshold_mask(heatmaps, threshold)
            pred_thresh = pred  # no threshold on pred for metrics

            heatmap_mse.append(metrics_wrapper.compute_mse(gt_thresh, pred_thresh))
            heatmap_ms_ssim.append(metrics_wrapper.compute_ms_ssim(gt_thresh, pred_thresh))
            heatmap_pixel_corr.extend(calculate_individual_pixel_correlation(gt_thresh, pred_thresh))

            # WFM: heatmap as magnitude (fx=magnitude, fy=0)
            B, _, H, W = gt_thresh.shape
            gt_ff = torch.zeros(B, 2, H, W, device=device)
            pred_ff = torch.zeros(B, 2, H, W, device=device)
            gt_ff[:, 0], pred_ff[:, 0] = gt_thresh[:, 0], pred_thresh[:, 0]
            try:
                wfm_corr_mag.append(wfm_correlation(gt_ff, pred_ff, mode="magnitude"))
                wfm_rel_mag_err.append(wfm_relative_magnitude_error(gt_ff, pred_ff))
            except Exception:
                wfm_corr_mag.append(float('nan'))
                wfm_rel_mag_err.append(float('nan'))

            force_sum_gt.extend(torch.sum(gt_thresh, dim=[1, 2, 3]).cpu().numpy().tolist())
            force_sum_pred.extend(torch.sum(pred_thresh, dim=[1, 2, 3]).cpu().numpy().tolist())
            force_mean_gt.extend(torch.mean(gt_thresh, dim=[1, 2, 3]).cpu().numpy().tolist())
            force_mean_pred.extend(torch.mean(pred_thresh, dim=[1, 2, 3]).cpu().numpy().tolist())

            if save_predictions:
                for i in range(images.size(0)):
                    p, t = pred_thresh[i:i+1], gt_thresh[i:i+1]
                    gt_ff_i = torch.zeros(1, 2, H, W, device=device)
                    pred_ff_i = torch.zeros(1, 2, H, W, device=device)
                    gt_ff_i[0, 0], pred_ff_i[0, 0] = t[0, 0], p[0, 0]
                    try:
                        rme = wfm_relative_magnitude_error(gt_ff_i, pred_ff_i)
                    except Exception:
                        rme = float('nan')
                    individual_predictions.append({
                        'batch_idx': batch_idx,
                        'sample_idx': i,
                        'original_image': images[i].cpu().numpy(),
                        'ground_truth': heatmaps[i].cpu().numpy(),
                        'ground_truth_thresholded': gt_thresh[i].cpu().numpy(),
                        'prediction': pred[i].cpu().numpy(),
                        'prediction_thresholded': pred_thresh[i].cpu().numpy(),
                        'mse': metrics_wrapper.compute_mse(t, p),
                        'ms_ssim': metrics_wrapper.compute_ms_ssim(t, p),
                        'pixel_correlation': calculate_pearson_correlation(t, p),
                        'wfm_relative_magnitude_error': rme,
                        'force_sum_gt': torch.sum(gt_thresh[i]).item(),
                        'force_sum_pred': torch.sum(pred_thresh[i]).item(),
                        'force_mean_gt': torch.mean(gt_thresh[i]).item(),
                        'force_mean_pred': torch.mean(pred_thresh[i]).item(),
                    })

    valid_wfm_corr = [x for x in wfm_corr_mag if not np.isnan(x)]
    valid_wfm_rme = [x for x in wfm_rel_mag_err if not np.isnan(x)]
    try:
        force_sum_corr, _ = pearsonr(force_sum_gt, force_sum_pred)
        force_mean_corr, _ = pearsonr(force_mean_gt, force_mean_pred)
    except Exception:
        force_sum_corr = force_mean_corr = 0.0
    if force_sum_corr is None or (isinstance(force_sum_corr, float) and np.isnan(force_sum_corr)):
        force_sum_corr = 0.0
    if force_mean_corr is None or (isinstance(force_mean_corr, float) and np.isnan(force_mean_corr)):
        force_mean_corr = 0.0

    results = {
        'heatmap': {
            'mse': np.mean(heatmap_mse),
            'mse_std': np.std(heatmap_mse),
            'ms_ssim': np.mean(heatmap_ms_ssim),
            'ms_ssim_std': np.std(heatmap_ms_ssim),
            'pixel_correlation': np.mean(heatmap_pixel_corr),
            'pixel_correlation_std': np.std(heatmap_pixel_corr),
        },
        'wfm': {
            'correlation_magnitude': np.mean(valid_wfm_corr) if valid_wfm_corr else float('nan'),
            'correlation_magnitude_std': np.std(valid_wfm_corr) if valid_wfm_corr else float('nan'),
            'relative_magnitude_error': np.mean(valid_wfm_rme) if valid_wfm_rme else float('nan'),
            'relative_magnitude_error_std': np.std(valid_wfm_rme) if valid_wfm_rme else float('nan'),
        },
        'force_sum': {
            'correlation': float(force_sum_corr),
            'gt_mean': np.mean(force_sum_gt),
            'pred_mean': np.mean(force_sum_pred),
            'gt_std': np.std(force_sum_gt),
            'pred_std': np.std(force_sum_pred),
        },
        'force_mean': {
            'correlation': float(force_mean_corr),
            'gt_mean': np.mean(force_mean_gt),
            'pred_mean': np.mean(force_mean_pred),
        },
    }

    if save_predictions:
        results['individual_predictions'] = individual_predictions
    return results


def print_metrics_report(report, threshold=0.0, uses_tanh=False):
    """Print formatted metrics report."""
    for name, metrics in report.items():
        print(f"\n🔸 {name.upper()} SET METRICS" + (f" (threshold={threshold})" if threshold > 0 else ""))
        print("-" * 60)
        print("HEATMAP METRICS:")
        print(f" MSE:             {metrics['heatmap']['mse']:.6f} ± {metrics['heatmap']['mse_std']:.6f}")
        print(f" MS-SSIM:         {metrics['heatmap']['ms_ssim']:.4f} ± {metrics['heatmap']['ms_ssim_std']:.4f}")
        print(f" Pixel Corr:      {metrics['heatmap']['pixel_correlation']:.4f} ± {metrics['heatmap']['pixel_correlation_std']:.4f}")
        print(f" Correlation (Magnitude): {metrics['wfm']['correlation_magnitude']:.4f} ± {metrics['wfm']['correlation_magnitude_std']:.4f}")
        print(f" Relative Magnitude Error: {metrics['wfm']['relative_magnitude_error']:.4f} ± {metrics['wfm']['relative_magnitude_error_std']:.4f}")
        print("FORCE SUM CORRELATION:")
        print(f" Correlation: {metrics['force_sum']['correlation']:.4f}")
        print(f" GT Mean: {metrics['force_sum']['gt_mean']:.2f} ± {metrics['force_sum']['gt_std']:.2f}")
        print(f" Pred Mean: {metrics['force_sum']['pred_mean']:.2f} ± {metrics['force_sum']['pred_std']:.2f}")
        if uses_tanh:
            print(" Note: Model outputs [-1,1], converted to [0,1] for evaluation")
    print("=" * 60)


def gen_prediction_plots(individual_predictions, save_dir, sort_by='ms_ssim', sort_order='desc', threshold=0.0):
    """Generate prediction plots (BF | GT | Pred) sorted by metric."""
    os.makedirs(save_dir, exist_ok=True)
    reverse = (sort_order.lower() == 'desc') if sort_by.lower() not in ['mse', 'wfm_relative_magnitude_error'] else (sort_order.lower() == 'desc')
    valid = [p for p in individual_predictions if not np.isnan(p.get(sort_by.lower(), 0))]
    sorted_preds = sorted(valid, key=lambda x: x[sort_by.lower()], reverse=reverse)
    print(f"Sorting {len(sorted_preds)} predictions by {sort_by} ({sort_order})")
    for rank, p in enumerate(tqdm(sorted_preds, desc="Saving plots"), 1):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        img = p['original_image']
        axes[0].imshow(img[0] if img.ndim == 3 else img, cmap='gray')
        axes[0].set_title('Bright Field')
        axes[0].axis('off')
        gt = p['ground_truth']
        axes[1].imshow(gt[0] if gt.ndim == 3 else gt, cmap='jet', vmin=0, vmax=1)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        pr = p['prediction']
        axes[2].imshow(pr[0] if pr.ndim == 3 else pr, cmap='jet', vmin=0, vmax=1)
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        m = (f"MSE: {p['mse']:.4f} | MS-SSIM: {p['ms_ssim']:.4f} | "
             f"Pixel Corr: {p['pixel_correlation']:.4f} | Rel Mag Err: {p.get('wfm_relative_magnitude_error', 'N/A')}")
        fig.suptitle(f"Rank {rank} (by {sort_by})\n{m}", fontsize=10, y=0.02)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"rank{rank:03d}_batch{p['batch_idx']:03d}_sample{p['sample_idx']:02d}.png"), dpi=150, bbox_inches='tight')
        plt.close()


def plot_predictions(loader, generator, n_samples, device, threshold=0.0,
                     use_settings=False, normalization_params=None, config_path=None, substrate_override=None):
    """Plot BF | GT | Pred for first n_samples from loader."""
    generator = generator.to(device)
    generator.eval()
    bf_list, gt_list, meta_list = [], [], []
    it = iter(loader)
    while len(bf_list) < n_samples:
        try:
            batch = next(it)
        except StopIteration:
            break
        if len(batch) == 5:
            images, heatmaps, _, _, meta = batch
        else:
            images, heatmaps = batch[0], batch[1]
            meta = None
        for i in range(images.shape[0]):
            if len(bf_list) >= n_samples:
                break
            bf_list.append(images[i])
            gt_list.append(heatmaps[i])
            meta_list.append(meta)
    n = min(n_samples, len(bf_list))
    bf_batch = torch.stack(bf_list[:n]).to(device, dtype=torch.float32)
    if use_settings and normalization_params:
        from models.s2f_model import create_settings_channels
        sub = substrate_override or 'fibroblasts_PDMS'
        meta_dict = {'substrate': [sub] * n}
        settings_ch = create_settings_channels(meta_dict, normalization_params, device, bf_batch.shape, config_path=config_path)
        bf_batch = torch.cat([bf_batch, settings_ch], dim=1)
    with torch.no_grad():
        pred = generator(bf_batch)
        if detect_tanh_output_model(generator):
            pred = convert_tanh_to_sigmoid_range(pred)
    if threshold > 0:
        pred = pred * (pred >= threshold).float()
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    for i in range(n):
        axes[i, 0].imshow(bf_list[i].squeeze().cpu().numpy(), cmap='gray')
        axes[i, 0].set_title('Bright Field')
        axes[i, 0].axis('off')
        axes[i, 1].imshow(gt_list[i].squeeze().cpu().numpy(), cmap='jet', vmin=0, vmax=1)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        axes[i, 2].imshow(pred[i].squeeze().cpu().numpy(), cmap='jet', vmin=0, vmax=1)
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    plt.tight_layout()
    plt.show()
