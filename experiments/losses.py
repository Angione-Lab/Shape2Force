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
