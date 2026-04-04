"""
S2F training logic: loss, metrics, and training loop.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Add S2F root to path
S2F_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if S2F_ROOT not in sys.path:
    sys.path.insert(0, S2F_ROOT)

from models.s2f_model import create_settings_channels
from utils.substrate_settings import compute_settings_normalization
from utils.metrics import (
    WFMRMELoss,
    calculate_psnr,
    calculate_ssim_tensor,
    calculate_pearson_correlation,
)
from scipy.stats import pearsonr


class S2FLoss(nn.Module):
    """S2F loss: reconstruction (WFM-RME by default) + GAN + optional force consistency."""
    def __init__(self, lambda_L1=100.0, lambda_gan=1.0, lambda_force=1.0,
                 gan_mode='vanilla', custom_loss=None, use_force_consistency=False,
                 force_consistency_target='mean'):
        super().__init__()
        self.lambda_L1 = lambda_L1
        self.lambda_gan = lambda_gan
        self.lambda_force = lambda_force
        self.gan_mode = gan_mode
        self.use_force_consistency = use_force_consistency
        self.force_consistency_target = force_consistency_target
        self.reconstruction_loss = custom_loss if custom_loss is not None else WFMRMELoss()
        self.force_consistency_loss = nn.MSELoss() if use_force_consistency else None
        self.gan_loss = nn.BCEWithLogitsLoss() if gan_mode == 'vanilla' else nn.MSELoss()

    def forward(self, pred, target, disc_pred=None, disc_target=None):
        recon_loss = self.reconstruction_loss(pred, target)
        gan_loss = 0.0
        if disc_pred is not None and disc_target is not None:
            gan_loss = self.gan_loss(disc_pred, disc_target)
        force_loss = 0.0
        if self.use_force_consistency and self.force_consistency_loss is not None:
            if self.force_consistency_target == 'mean':
                pred_global = torch.mean(pred.view(pred.size(0), -1), dim=1, keepdim=True)
                target_global = torch.mean(target.view(target.size(0), -1), dim=1, keepdim=True)
            else:
                pred_global = torch.sum(pred.view(pred.size(0), -1), dim=1, keepdim=True)
                target_global = torch.sum(target.view(target.size(0), -1), dim=1, keepdim=True)
            force_loss = self.force_consistency_loss(pred_global, target_global)
        total = self.lambda_L1 * recon_loss + self.lambda_gan * gan_loss + self.lambda_force * force_loss
        return total, recon_loss, gan_loss, force_loss


def calculate_soft_dice_loss(pred, target, smooth=1e-6):
    """Dice score (higher is better)."""
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    dice_scores = (2.0 * intersection + smooth) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth)
    return dice_scores.mean().item()


def train_s2f(generator, discriminator, train_loader, val_loader, device='cuda',
              num_epochs=100, g_lr=2e-4, d_lr=2e-4, beta1=0.5, beta2=0.999,
              save_dir='ckp', lambda_L1=100.0, lambda_gan=1.0, lambda_force=1.0,
              gan_mode='vanilla', save_predictions_every=5, custom_loss=None,
              loaded_metadata=False, use_settings=False, use_force_consistency=False,
              force_consistency_target='mean', config_path=None):
    """
    Train S2F model.
    """
    from diffusers.optimization import get_cosine_schedule_with_warmup

    config_path = config_path or os.path.join(S2F_ROOT, 'config', 'substrate_settings.json')
    normalization_params = None
    if use_settings:
        if not loaded_metadata:
            raise ValueError("loaded_metadata must be True when use_settings=True")
        normalization_params = compute_settings_normalization(config_path=config_path)

    history = {'g_loss': [], 'd_loss': [], 'g_recon_loss': [], 'g_gan_loss': [], 'g_force_loss': [],
               'train_loss': [], 'val_loss': [], 'train_ssim': [], 'val_ssim': [],
               'train_psnr': [], 'val_psnr': [], 'train_mse': [], 'val_mse': [],
               'train_dice_score': [], 'val_dice_score': []}

    if not torch.cuda.is_available() and device == 'cuda':
        device = 'cpu'
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    criterion = S2FLoss(lambda_L1=lambda_L1, lambda_gan=lambda_gan, lambda_force=lambda_force,
                        gan_mode=gan_mode, custom_loss=custom_loss,
                        use_force_consistency=use_force_consistency,
                        force_consistency_target=force_consistency_target)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=g_lr, betas=(beta1, beta2))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=d_lr, betas=(beta1, beta2))
    num_steps = len(train_loader) * num_epochs
    g_scheduler = get_cosine_schedule_with_warmup(g_optimizer, int(num_steps * 0.1), num_steps)
    d_scheduler = get_cosine_schedule_with_warmup(d_optimizer, int(num_steps * 0.1), num_steps)

    os.makedirs(save_dir, exist_ok=True)
    vis_dir = os.path.join(save_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    best_val_loss = float('inf')
    disc_output_shape = None

    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        g_loss_total = d_loss_total = g_recon_total = g_gan_total = g_force_total = 0.0
        train_ssim = train_psnr = train_mse = train_dice = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

        for batch_data in pbar:
            if loaded_metadata:
                input_images, target_images, _, _, metadata = batch_data
            else:
                input_images, target_images, _, _ = batch_data
            input_images = input_images.to(device, dtype=torch.float32)
            target_images = target_images.to(device, dtype=torch.float32)
            batch_size = input_images.size(0)

            if use_settings and normalization_params is not None:
                settings_channels = create_settings_channels(
                    metadata, normalization_params, device, input_images.shape,
                    config_path=config_path
                )
                input_images = torch.cat([input_images, settings_channels], dim=1)

            target_scaled = target_images * 2.0 - 1.0
            if disc_output_shape is None:
                with torch.no_grad():
                    dummy = torch.cat([input_images[:1], target_scaled[:1]], dim=1)
                    disc_output_shape = discriminator(dummy).shape[2:]
            real_labels = torch.ones(batch_size, 1, *disc_output_shape).to(device)
            fake_labels = torch.zeros(batch_size, 1, *disc_output_shape).to(device)

            g_optimizer.zero_grad()
            fake_images = generator(input_images)
            fake_for_loss = (fake_images + 1.0) / 2.0
            fake_input = torch.cat([input_images, fake_images], dim=1)
            fake_pred = discriminator(fake_input)
            g_loss, g_recon, g_gan, g_force = criterion(fake_for_loss, target_images, fake_pred, real_labels)
            g_loss.backward()
            g_optimizer.step()

            d_optimizer.zero_grad()
            real_input = torch.cat([input_images, target_scaled], dim=1)
            real_pred = discriminator(real_input)
            d_real = criterion.gan_loss(real_pred, real_labels)
            fake_input_d = torch.cat([input_images, fake_images.detach()], dim=1)
            fake_pred_d = discriminator(fake_input_d)
            d_fake = criterion.gan_loss(fake_pred_d, fake_labels)
            d_loss = (d_real + d_fake) * 0.5
            d_loss.backward()
            d_optimizer.step()
            g_scheduler.step()
            d_scheduler.step()

            g_loss_total += g_loss.item()
            d_loss_total += d_loss.item()
            g_recon_total += g_recon.item()
            g_gan_total += g_gan.item()
            g_force_total += g_force.item() if isinstance(g_force, torch.Tensor) else g_force
            train_ssim += calculate_ssim_tensor(fake_for_loss, target_images)
            train_psnr += calculate_psnr(fake_for_loss, target_images)
            train_mse += F.mse_loss(fake_for_loss, target_images).item()
            train_dice += calculate_soft_dice_loss(fake_for_loss, target_images)
            pbar.set_postfix({'G': g_loss.item(),
                'D': d_loss.item(), 'Dice': train_dice / (pbar.n + 1)})

        n_train = len(train_loader)
        g_loss_total /= n_train
        d_loss_total /= n_train
        train_ssim /= n_train
        train_psnr /= n_train
        train_mse /= n_train
        train_dice /= n_train

        generator.eval()
        val_loss = val_ssim = val_psnr = val_mse = val_dice = 0.0
        with torch.no_grad():
            for batch_data in val_loader:
                if loaded_metadata:
                    input_images, target_images, _, _, metadata = batch_data
                else:
                    input_images, target_images, _, _ = batch_data
                input_images = input_images.to(device, dtype=torch.float32)
                target_images = target_images.to(device, dtype=torch.float32)
                if use_settings and normalization_params is not None:
                    settings_channels = create_settings_channels(
                        metadata, normalization_params, device, input_images.shape,
                        config_path=config_path
                    )
                    input_images = torch.cat([input_images, settings_channels], dim=1)
                fake_images = generator(input_images)
                fake_for_loss = (fake_images + 1.0) / 2.0
                _, recon_loss, _, force_loss = criterion(fake_for_loss, target_images)
                val_loss += recon_loss.item()
                val_ssim += calculate_ssim_tensor(fake_for_loss, target_images)
                val_psnr += calculate_psnr(fake_for_loss, target_images)
                val_mse += F.mse_loss(fake_for_loss, target_images).item()
                val_dice += calculate_soft_dice_loss(fake_for_loss, target_images)
        n_val = len(val_loader)
        val_loss /= n_val
        val_ssim /= n_val
        val_psnr /= n_val
        val_mse /= n_val
        val_dice /= n_val

        history['g_loss'].append(g_loss_total)
        history['d_loss'].append(d_loss_total)
        history['train_loss'].append(g_loss_total)
        history['val_loss'].append(val_loss)
        history['train_ssim'].append(train_ssim)
        history['val_ssim'].append(val_ssim)
        history['train_psnr'].append(train_psnr)
        history['val_psnr'].append(val_psnr)
        history['train_mse'].append(train_mse)
        history['val_mse'].append(val_mse)
        history['train_dice_score'].append(train_dice)
        history['val_dice_score'].append(val_dice)

        best_mark = "✓" if val_loss < best_val_loss else ""
        print(f"Train: G_Loss:{g_loss_total:.4f} D_Loss:{d_loss_total:.4f} "
              f"MSE:{train_mse:.4f} SSIM:{train_ssim:.4f} Dice:{train_dice:.4f}")
        print(f"Valid: Loss:{val_loss:.4f} MSE:{val_mse:.4f} SSIM:{val_ssim:.4f} Dice:{val_dice:.4f} {best_mark}")

        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'g_optimizer_state_dict': g_optimizer.state_dict(),
            'd_optimizer_state_dict': d_optimizer.state_dict(),
            'val_loss': val_loss,
            'history': history
        }
        torch.save(checkpoint, os.path.join(save_dir, 'last_checkpoint.pth'))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(save_dir, 'best_checkpoint.pth'))

        if epoch % save_predictions_every == 0:
            generator.eval()
            with torch.no_grad():
                batch_data = next(iter(val_loader))
                if loaded_metadata:
                    input_images, target_images, _, _, metadata = batch_data
                else:
                    input_images, target_images, _, _ = batch_data
                input_images = input_images.to(device, dtype=torch.float32)
                target_images = target_images.to(device, dtype=torch.float32)
                if use_settings and normalization_params is not None:
                    settings_channels = create_settings_channels(
                        metadata, normalization_params, device, input_images.shape,
                        config_path=config_path
                    )
                    input_images = torch.cat([input_images, settings_channels], dim=1)
                fake_images = generator(input_images)
                fake_vis = (fake_images + 1.0) / 2.0
                n_vis = min(4, input_images.size(0))
                fig, axes = plt.subplots(3, n_vis, figsize=(4 * n_vis, 12))
                if n_vis == 1:
                    axes = axes.reshape(3, 1)
                for i in range(n_vis):
                    axes[0, i].imshow(input_images[i, 0].cpu().numpy(), cmap='gray')
                    axes[0, i].axis('off')
                    axes[1, i].imshow(fake_vis[i, 0].cpu().numpy(), cmap='jet', vmin=0, vmax=1)
                    axes[1, i].axis('off')
                    axes[2, i].imshow(target_images[i, 0].cpu().numpy(), cmap='jet', vmin=0, vmax=1)
                    axes[2, i].axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, f'predictions_epoch_{epoch:02d}.png'), dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Saved visualization for epoch {epoch}")

    return history
