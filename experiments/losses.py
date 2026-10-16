"""
Reconstruction loss used for the released single-cell model.

Per-image weighted force-magnitude relative error (WFM-RME), as in the manuscript and as in the
loss that trained the released checkpoint (Archive/src/utils/losses.WFMRelativeMagnitudeLoss):
the weight of each pixel is its ground-truth magnitude divided by the mean ground-truth magnitude
*of the same image* (eps = 1e-6).  Note: utils.metrics.WFMRMELoss normalises by the mean over the
whole batch instead; the two coincide only for batch size 1.
"""
import torch
import torch.nn as nn


class PerImageWFMRMELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        t, p = target[:, 0], pred[:, 0]                       # (B, H, W) magnitudes
        rel = (p - t).abs() / (t + self.eps)
        w = t / (t.mean(dim=(1, 2), keepdim=True) + self.eps)
        return (rel * w).mean()


class WeightedMSEPlusDiceLoss(nn.Module):
    """Loss of the image-only training stage of the released model (Archive/src/utils/losses.py,
    used in Archive/notebooks/train_pix2pix.ipynb with pos_weight=3, threshold=0.2, dice_weight=0.5).
    Pixels above `threshold` get weight 1 + pos_weight in the MSE; the soft-Dice term rewards overlap of
    the predicted and true high-force regions. Unlike L1 / WFM-RME it drives the network to localise
    the force-generating regions when training starts from random weights."""
    def __init__(self, pos_weight=3.0, threshold=0.2, dice_weight=0.5):
        super().__init__()
        self.pos_weight, self.threshold, self.dice_weight = pos_weight, threshold, dice_weight

    def forward(self, pred, target):
        w = 1.0 + self.pos_weight * (target > self.threshold).float()
        mse = (w * (pred - target) ** 2).mean()
        p, t = pred.flatten(1), target.flatten(1)
        dice = (2 * (p * t).sum(1) + 1e-6) / (p.sum(1) + t.sum(1) + 1e-6)
        return mse + self.dice_weight * (1 - dice.mean())


REC_LOSSES = {"rme": PerImageWFMRMELoss, "wmd": WeightedMSEPlusDiceLoss}
