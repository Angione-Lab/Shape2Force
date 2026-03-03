"""
S2F (Shape2Force) model for force map prediction (inference only).
Supports single-cell and spheroid modes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ResidualBlock
from .cbam import CBAM

from utils.substrate_settings import get_settings_of_category


def normalize_settings(substrate_name, normalization_params, config=None, config_path=None):
    """
    Normalize settings for a given substrate.

    Args:
        substrate_name (str): Name of the substrate
        normalization_params (dict): Normalization parameters

    Returns:
        tuple: (normalized_pixelsize, normalized_young)
    """
    settings = get_settings_of_category(substrate_name, config=config, config_path=config_path)

    # Min-max normalization to [0, 1]
    pixelsize_norm = (settings['pixelsize'] - normalization_params['pixelsize']['min']) / \
                     (normalization_params['pixelsize']['max'] - normalization_params['pixelsize']['min'])

    young_norm = (settings['young'] - normalization_params['young']['min']) / \
                 (normalization_params['young']['max'] - normalization_params['young']['min'])

    return pixelsize_norm, young_norm

def create_settings_channels(metadata, normalization_params, device, image_shape, config_path=None):
    """
    Create settings channels for a batch of images.

    Args:
        metadata (dict): Batch metadata containing substrate information
        normalization_params (dict): Normalization parameters
        device: Device to create tensors on
        image_shape (tuple): Shape of input images (B, C, H, W)

    Returns:
        torch.Tensor: Settings channels [B, 2, H, W] where channels are [pixelsize, young]
    """
    batch_size, _, height, width = image_shape

    # Create settings channels
    pixelsize_channel = torch.zeros(batch_size, 1, height, width, device=device)
    young_channel = torch.zeros(batch_size, 1, height, width, device=device)

    for i in range(batch_size):
        substrate = metadata['substrate'][i]
        pixelsize_norm, young_norm = normalize_settings(
            substrate, normalization_params, config_path=config_path
        )

        # Fill entire channel with normalized value
        pixelsize_channel[i, 0] = pixelsize_norm
        young_channel[i, 0] = young_norm

    # Concatenate channels
    settings_channels = torch.cat([pixelsize_channel, young_channel], dim=1)  # [B, 2, H, W]

    return settings_channels

class GlobalContextModule(nn.Module):
    """A module for capturing cell shape information"""
    def __init__(self, in_channels):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//4, in_channels, 1),
            nn.Sigmoid()
        )
        self.large_kernel = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.multi_scale = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels//4, 3, padding=1, dilation=1),
            nn.Conv2d(in_channels, in_channels//4, 3, padding=2, dilation=2),
            nn.Conv2d(in_channels, in_channels//4, 3, padding=4, dilation=4),
            nn.Conv2d(in_channels, in_channels//4, 3, padding=8, dilation=8)
        ])
        self.fusion = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        global_ctx = self.global_pool(x)
        global_weight = self.global_conv(global_ctx)
        large_features = self.large_kernel(x)
        multi_scale_features = []
        for conv in self.multi_scale:
            multi_scale_features.append(conv(x))
        multi_scale_out = torch.cat(multi_scale_features, dim=1)
        multi_scale_out = self.fusion(multi_scale_out)
        return x + (large_features * global_weight) + multi_scale_out

class HierarchicalAttention(nn.Module):
    """A module for combining spatial and channel attention"""
    def __init__(self, channels):
        super().__init__()
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, channels//8, 1),
            nn.Conv2d(channels//8, 1, 3, padding=1),
            nn.Sigmoid()
        )
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//16, channels, 1),
            nn.Sigmoid()
        )
        self.cross_att = nn.Sequential(
            nn.Conv2d(channels, channels//4, 1),
            nn.BatchNorm2d(channels//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//4, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        spatial_weight = self.spatial_att(x)
        channel_weight = self.channel_att(x)
        attended = x * spatial_weight * channel_weight
        cross_weight = self.cross_att(attended)
        return x + (attended * cross_weight)

class AttentionGate(nn.Module):
    """Attention gate with global context"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(F_int, F_int//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(F_int//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(F_int//2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(F_l, F_int//4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(F_int//4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        psi = self.psi(g1 + x1)
        global_weight = self.global_context(x)
        psi = psi * global_weight
        if psi.shape[2:] != x.shape[2:]:
            psi = F.interpolate(psi, size=x.shape[2:], mode='bilinear', align_corners=False)
        return x * psi

class SpheroidAttentionGate(nn.Module):
    """Attention Gate from ForceNet2WithAttention (s2f_spheroid). Checkpoint-compatible for ckp_spheroid_FN.pth."""
    def __init__(self, F_g, F_l, F_int):
        super(SpheroidAttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.psi(g1 + x1)
        return x * psi

class PatchGANDiscriminator(nn.Module):
    """PatchGAN Discriminator (included for create_s2f_model compatibility)."""
    def __init__(self, in_channels=2, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super().__init__()
        use_bias = norm_layer == nn.InstanceNorm2d
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1, bias=use_bias),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layers = nn.ModuleList()
        nf_mult, nf_mult_prev = 1, 1
        for n in range(1, n_layers):
            nf_mult_prev, nf_mult = nf_mult, min(2 ** n, 8)
            self.layers.append(nn.Sequential(
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ))
        nf_mult_prev, nf_mult = nf_mult, min(2 ** n_layers, 8)
        self.layers.append(nn.Sequential(
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True)
        ))
        self.output_conv = nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        self.attention = nn.Sequential(
            nn.Conv2d(ndf * nf_mult, ndf * nf_mult // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ndf * nf_mult // 4, ndf * nf_mult, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        x = self.initial_conv(input)
        for layer in self.layers:
            x = layer(x)
        x = x * self.attention(x)
        return self.output_conv(x)

class S2FGenerator(nn.Module):
    """
    S2F (Shape2Force) model: U-Net generator for force map prediction.
    Supports substrate-specific settings as additional input channels.
    """
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 img_size=1024,
                 bridge_type='cbam',
                 use_multi_scale_input=True):
        super().__init__()

        self.img_size = img_size
        self.bridge_type = bridge_type
        self.use_multi_scale_input = use_multi_scale_input

        if self.use_multi_scale_input:
            self.scale_pyramid = nn.ModuleList([
                nn.Conv2d(in_channels, 32, 3, padding=1),
                nn.Sequential(
                    nn.AvgPool2d(2, stride=2),
                    nn.Conv2d(in_channels, 32, 3, padding=1)
                ),
                nn.Sequential(
                    nn.AvgPool2d(4, stride=4),
                    nn.Conv2d(in_channels, 32, 3, padding=1)
                )
            ])
            self.initial_conv = nn.Conv2d(96, 64, 1)
        else:
            self.initial_conv = nn.Conv2d(in_channels, 64, 3, padding=1)

        def reg_conv_block(in_c, out_c, use_attention=True):
            layers = [
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                ResidualBlock(out_c, out_c)
            ]
            if use_attention:
                layers.append(HierarchicalAttention(out_c))
            return nn.Sequential(*layers)

        def dilated_conv_block(in_c, out_c, use_global_context=False):
            layers = [
                nn.Conv2d(in_c, out_c, 3, padding=2, dilation=2),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                ResidualBlock(out_c, out_c)
            ]
            if use_global_context:
                layers.append(GlobalContextModule(out_c))
            return nn.Sequential(*layers)

        self.encoder1 = reg_conv_block(64, 64, use_attention=False)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = reg_conv_block(64, 128, use_attention=True)
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = dilated_conv_block(128, 256, use_global_context=True)
        self.pool3 = nn.MaxPool2d(2)
        self.encoder4 = dilated_conv_block(256, 512, use_global_context=True)
        self.pool4 = nn.MaxPool2d(2)

        if bridge_type == 'cbam':
            self.bridge = nn.Sequential(
                dilated_conv_block(512, 1024, use_global_context=True),
                CBAM(1024),
                GlobalContextModule(1024),
                HierarchicalAttention(1024)
            )
        else:
            self.bridge = nn.Sequential(
                dilated_conv_block(512, 1024, use_global_context=True),
                GlobalContextModule(1024),
                HierarchicalAttention(1024)
            )

        self.att4 = AttentionGate(512, 512, 256)
        self.att3 = AttentionGate(256, 256, 128)
        self.att2 = AttentionGate(128, 128, 64)
        self.att1 = AttentionGate(64, 64, 32)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = reg_conv_block(1024, 512, use_attention=True)
        self.refine4 = HierarchicalAttention(512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = reg_conv_block(512, 256, use_attention=True)
        self.refine3 = HierarchicalAttention(256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = reg_conv_block(256, 128, use_attention=True)
        self.refine2 = HierarchicalAttention(128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = reg_conv_block(128, 64, use_attention=True)
        self.refine1 = HierarchicalAttention(64)

        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 1),
            nn.Tanh()
        )

    def forward(self, x):
        if self.use_multi_scale_input:
            scale_features = []
            for i, scale_conv in enumerate(self.scale_pyramid):
                if i == 0:
                    scale_features.append(scale_conv(x))
                else:
                    scale_out = scale_conv(x)
                    scale_out = F.interpolate(scale_out, size=x.shape[2:], mode='bilinear', align_corners=False)
                    scale_features.append(scale_out)
            fused = torch.cat(scale_features, dim=1)
            initial_features = self.initial_conv(fused)
        else:
            initial_features = self.initial_conv(x)

        e1 = self.encoder1(initial_features)
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(self.pool2(e2))
        e4 = self.encoder4(self.pool3(e3))
        b = self.bridge(self.pool4(e4))

        g4 = self.up4(b)
        x4 = self.att4(g4, e4)
        d4 = self.dec4(torch.cat([g4, x4], dim=1))
        d4 = self.refine4(d4)
        g3 = self.up3(d4)
        x3 = self.att3(g3, e3)
        d3 = self.dec3(torch.cat([g3, x3], dim=1))
        d3 = self.refine3(d3)
        g2 = self.up2(d3)
        x2 = self.att2(g2, e2)
        d2 = self.dec2(torch.cat([g2, x2], dim=1))
        d2 = self.refine2(d2)
        g1 = self.up1(d2)
        x1 = self.att1(g1, e1)
        d1 = self.dec1(torch.cat([g1, x1], dim=1))
        d1 = self.refine1(d1)
        out = self.final_conv(d1)
        return out


class S2FSpheroidGenerator(nn.Module):
    """
    A s2f model with some tunings for spheroid data
    """
    def __init__(self, in_channels=1, out_channels=1, predict_numbers=False, img_size=1024, use_tanh_output=True):
        super(S2FSpheroidGenerator, self).__init__()
        self.predict_numbers = predict_numbers
        self.img_size = img_size
        self.use_tanh_output = use_tanh_output

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                ResidualBlock(out_c, out_c)
            )

        # Encoder
        self.encoder1 = conv_block(in_channels, 32)  # [B, 32, 1024, 1024]
        self.pool1 = nn.MaxPool2d(2)  # [B, 32, 512, 512]
        self.encoder2 = conv_block(32, 64)  # [B, 64, 512, 512]
        self.pool2 = nn.MaxPool2d(2)  # [B, 64, 256, 256]
        self.encoder3 = conv_block(64, 128)  # [B, 128, 256, 256]
        self.pool3 = nn.MaxPool2d(2)  # [B, 128, 128, 128]
        self.encoder4 = conv_block(128, 256)  # [B, 256, 128, 128]
        self.pool4 = nn.MaxPool2d(2)  # [B, 256, 64, 64]
        self.bridge = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            ResidualBlock(512, 512)
        )  # [B, 512, 64, 64]

        # Attention Gates (SpheroidAttentionGate from s2f_spheroid, matches ckp_spheroid_FN.pth)
        self.att3 = SpheroidAttentionGate(256, 256, 128)
        self.att2 = SpheroidAttentionGate(128, 128, 64)
        self.att1 = SpheroidAttentionGate(64, 64, 32)

        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # [B, 256, 128, 128]
        self.dec3 = conv_block(512, 256)  # [B, 256, 128, 128]
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # [B, 128, 256, 256]
        self.dec2 = conv_block(256, 128)  # [B, 128, 256, 256]
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)   # [B, 64, 512, 512]
        self.dec1 = conv_block(128, 64)   # [B, 64, 512, 512]
        self.up0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)    # [B, 32, 1024, 1024]
        self.dec0 = conv_block(64, 32)    # [B, 32, 1024, 1024]
        
        # Final prediction
        self.pred_conv = nn.Conv2d(32, out_channels, kernel_size=1)  # [B, 1, 1024, 1024]

    def forward(self, x):  # Input: [B, 1, 1024, 1024]
        # Encoder
        e1 = self.encoder1(x)            # [B, 32, 1024, 1024]
        e2 = self.encoder2(self.pool1(e1))  # [B, 64, 512, 512]
        e3 = self.encoder3(self.pool2(e2))  # [B, 128, 256, 256]
        e4 = self.encoder4(self.pool3(e3))  # [B, 256, 128, 128]
        b = self.bridge(self.pool4(e4))     # [B, 512, 64, 64]

        # Decoder + Attention
        g3 = self.up3(b)  # [B, 256, 128, 128]
        x3 = self.att3(g3, e4)  # [B, 256, 128, 128]
        d3 = self.dec3(torch.cat([g3, x3], dim=1))  # [B, 256, 128, 128]

        g2 = self.up2(d3)  # [B, 128, 256, 256]
        x2 = self.att2(g2, e3)  # [B, 128, 256, 256]
        d2 = self.dec2(torch.cat([g2, x2], dim=1))  # [B, 128, 256, 256]

        g1 = self.up1(d2)  # [B, 64, 512, 512]
        x1 = self.att1(g1, e2)  # [B, 64, 512, 512]
        d1 = self.dec1(torch.cat([g1, x1], dim=1))  # [B, 64, 512, 512]

        g0 = self.up0(d1)  # [B, 32, 1024, 1024]
        d0 = self.dec0(torch.cat([g0, e1], dim=1))  # [B, 32, 1024, 1024]
 
        out = self.pred_conv(d0)  # [B, 1, 1024, 1024]
        out_resized = F.interpolate(out, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        
        if self.use_tanh_output:
            return torch.tanh(out_resized)  # [-1, 1] for Pix2Pix training
        else:
            return torch.sigmoid(out_resized)  # [0, 1] for direct inference

    def predict(self, loader):
        """
        Predict on the first batch from the loader
        """
        self.eval()
        with torch.no_grad():
            # Get first batch from loader
            batch = next(iter(loader))
            input_images, ground_truth_heatmaps, _, _ = batch  # Ignore cell_area and cell_force
            
            # Move to same device as model
            device = next(self.parameters()).device
            input_images = input_images.to(device)
            ground_truth_heatmaps = ground_truth_heatmaps.to(device)
            
            # Get predictions
            predicted_heatmaps = self(input_images)
            
            if self.use_tanh_output:
                predicted_heatmaps = (predicted_heatmaps + 1.0) / 2.0
            
            return input_images, ground_truth_heatmaps, predicted_heatmaps

    
    def set_output_mode(self, use_tanh=True):
        """
        Set the output activation mode
        
        Args:
            use_tanh: If True, use tanh output [-1, 1] for GAN training
                     If False, use sigmoid output [0, 1] for direct inference
        """
        self.use_tanh_output = use_tanh
        if use_tanh:
            print("Generator set to tanh output mode [-1, 1] for GAN training")
        else:
            print("Generator set to sigmoid output mode [0, 1] for inference/evaluation")

def create_s2f_model(
    in_channels=1,
    out_channels=1,
    img_size=1024,
    bridge_type='cbam',
    use_multi_scale_input=True,
    ndf=64,
    n_layers=3,
    model_type='s2f',
):
    """Create S2F model with generator and discriminator."""
    if model_type == 's2f':
        generator = S2FGenerator(
        in_channels=in_channels,
        out_channels=out_channels,
        img_size=img_size,
        bridge_type=bridge_type,
        use_multi_scale_input=use_multi_scale_input,
        )
    elif model_type == 's2f_spheroid':
        generator = S2FSpheroidGenerator(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
        )
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    discriminator = PatchGANDiscriminator(
        in_channels=in_channels + out_channels,
        ndf=ndf,
        n_layers=n_layers
    )
    return generator, discriminator
