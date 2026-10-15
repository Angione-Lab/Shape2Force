"""
Generators for the baseline / ablation study. All variants output tanh in [-1, 1], like the
released S2F generator, so they share one training and evaluation path.

  arch='s2f'   S2FGenerator (models/s2f_model.py) with optional switches:
               use_multi_scale_input  (False -> single 3x3 input conv)
               bridge                 'cbam' (paper) | 'cnn' (no CBAM) | 'plain' (dilated conv block only:
                                      no CBAM, no Global Context Module, no hierarchical attention)
               attention_gates        False -> skip connections pass encoder features unchanged
  arch='unet'  plain U-Net (Ronneberger et al., 2015): 4 levels, 64-1024 channels, double 3x3 conv + BN +
               ReLU, transposed-conv upsampling, concatenation skips; no attention modules.
"""
import torch
import torch.nn as nn

from models.blocks import ResidualBlock
from models.s2f_model import PatchGANDiscriminator, S2FGenerator


class _PassSkip(nn.Module):
    """Replaces an attention gate: returns the encoder skip feature unchanged."""
    def forward(self, g, x):
        return x


def _double_conv(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base=64):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.downs = nn.ModuleList([_double_conv(in_channels, c[0])] + [_double_conv(c[i], c[i + 1]) for i in range(3)])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _double_conv(c[3], c[4])
        self.ups = nn.ModuleList([nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2) for i in reversed(range(4))])
        self.decs = nn.ModuleList([_double_conv(c[i] * 2, c[i]) for i in reversed(range(4))])
        self.out = nn.Sequential(nn.Conv2d(c[0], out_channels, 1), nn.Tanh())

    def forward(self, x):
        skips = []
        for d in self.downs:
            x = d(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, s in zip(self.ups, self.decs, reversed(skips)):
            x = dec(torch.cat([up(x), s], dim=1))
        return self.out(x)


def build_generator(arch="s2f", in_channels=3, use_multi_scale_input=True, bridge="cbam",
                    attention_gates=True, **_):
    if arch == "unet":
        return UNet(in_channels=in_channels)
    if arch != "s2f":
        raise ValueError(f"unknown arch {arch}")
    g = S2FGenerator(in_channels=in_channels, out_channels=1, img_size=1024,
                     bridge_type="cnn" if bridge in ("cnn", "plain") else "cbam",
                     use_multi_scale_input=use_multi_scale_input)
    if bridge == "plain":
        g.bridge = nn.Sequential(nn.Conv2d(512, 1024, 3, padding=2, dilation=2), nn.BatchNorm2d(1024),
                                 nn.ReLU(inplace=True), ResidualBlock(1024, 1024))
    if not attention_gates:
        g.att4, g.att3, g.att2, g.att1 = _PassSkip(), _PassSkip(), _PassSkip(), _PassSkip()
    return g


def build_discriminator(in_channels=3):
    return PatchGANDiscriminator(in_channels=in_channels + 1, ndf=64, n_layers=3)


def n_params(module):
    return sum(p.numel() for p in module.parameters())
