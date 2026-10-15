"""
Training protocol shared by every baseline / ablation run, and the variants.

All variants are trained FROM SCRATCH with the same protocol, split, number of epochs and
checkpoint-selection rule, so that differences are attributable to the ablated component only.
(The released single-cell checkpoint was obtained by staged fine-tuning, see the Methods; the
's2f_full' variant is its from-scratch counterpart under this protocol.)
"""
import copy

PROTOCOL = dict(
    train_manifest="alltune_train",      # 280 pairs, released model's training split
    val_manifest="alltune_val",          # 81 pairs, used for checkpoint selection only
    epochs=100,
    batch_size=2,
    img_size=1024,
    augment=True,                        # data/augmentations.AdvancedAugmentations
    g_lr=1e-4, d_lr=1e-4, beta1=0.5, beta2=0.999,
    warmup_frac=0.1,                     # cosine schedule with linear warm-up
    lambda_rec=10.0,                     # per-image WFM-RME
    lambda_gan=1.0,                      # vanilla (BCE) PatchGAN loss
    lambda_force=100.0,                  # MSE between mean predicted and mean target intensity
    arch="s2f", in_channels=3, use_multi_scale_input=True, bridge="cbam", attention_gates=True,
    substrate_mode="legacy",             # see experiments/manifest_data.py
    input_mode="bf",
    seed=0,
)

VARIANTS = {
    # reference
    "s2f_full":               dict(),
    # baselines
    "unet":                   dict(arch="unet", lambda_gan=0.0),          # plain U-Net, no attention, no discriminator
    "unet_gan":               dict(arch="unet"),                          # pix2pix-style U-Net + PatchGAN
    "mask_only":              dict(input_mode="mask"),                    # binary cell mask instead of bright field
    # loss ablations
    "s2f_no_gan":             dict(lambda_gan=0.0),
    "s2f_no_force":           dict(lambda_force=0.0),
    "s2f_gan_0.1":            dict(lambda_gan=0.1),
    "s2f_gan_5":              dict(lambda_gan=5.0),
    # input ablations
    "s2f_no_substrate":       dict(in_channels=1, substrate_mode="none"),
    "s2f_nominal_substrate":  dict(substrate_mode="nominal"),
    # architecture ablations
    "s2f_no_multiscale":      dict(use_multi_scale_input=False),
    "s2f_bridge_no_cbam":     dict(bridge="cnn"),
    "s2f_bridge_plain":       dict(bridge="plain"),
    "s2f_no_attention_gates": dict(attention_gates=False),
}

# Recommended order if GPU time is limited (about 7-8 h per run of 100 epochs on an 11 GB GPU)
PRIORITY = ["s2f_full", "unet", "s2f_no_gan", "mask_only", "s2f_no_substrate",
            "s2f_no_multiscale", "s2f_bridge_plain", "s2f_no_force", "s2f_no_attention_gates",
            "s2f_bridge_no_cbam", "s2f_nominal_substrate", "unet_gan", "s2f_gan_0.1", "s2f_gan_5"]


def get_config(variant, **overrides):
    if variant not in VARIANTS:
        raise KeyError(f"unknown variant '{variant}'. Options: {', '.join(VARIANTS)}")
    cfg = copy.deepcopy(PROTOCOL)
    cfg.update(VARIANTS[variant])
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    cfg["variant"] = variant
    if cfg["substrate_mode"] == "none":
        cfg["in_channels"] = 1
    return cfg
