"""
Training protocol shared by every baseline / ablation run, and the variants.

All variants are trained FROM SCRATCH with the same two-phase protocol, split, number of epochs and
checkpoint-selection rule, so that differences are attributable to the ablated component only.
The two phases mirror the staged training of the released model:
  phase 1  localisation:  weighted MSE + soft Dice (lambda_rec 100), lr 1e-4    (as train_pix2pix.ipynb)
  phase 2  magnitude:     per-image WFM-RME (lambda_rec 10) + force consistency (lambda_force 100),
                          lr 1e-5, initialised from the best phase-1 generator  (as train_pix2pix_settings_mean.ipynb)
WFM-RME alone does not learn to localise forces from random weights (tested: pixel r ~ 0 after 100 epochs),
which is why the released model, too, was first trained with the weighted MSE + Dice loss.
Unlike the released model, every variant sees its full input (incl. substrate channels) from the start.
"""
import copy

PROTOCOL = dict(
    train_manifest="alltune_train",      # 280 pairs, released model's training split
    val_manifest="alltune_val",          # 81 pairs, used for checkpoint selection only
    phases=[
        dict(name="phase1", epochs=100, rec_loss="wmd", lambda_rec=100.0, lambda_force=0.0, g_lr=1e-4, d_lr=1e-4),
        dict(name="phase2", epochs=40, rec_loss="rme", lambda_rec=10.0, lambda_force="from_variant", g_lr=1e-5, d_lr=1e-5),
    ],
    batch_size=2,
    img_size=1024,
    augment=True,                        # data/augmentations.AdvancedAugmentations
    beta1=0.5, beta2=0.999,
    warmup_frac=0.1,                     # cosine schedule with linear warm-up (per phase)
    lambda_gan=1.0,                      # vanilla (BCE) PatchGAN loss, both phases
    lambda_force=100.0,                  # phase 2: MSE between mean predicted and mean target intensity
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
    "s2f_no_force":           dict(lambda_force=0.0, share_phase1_with="s2f_full"),   # force term is a phase-2 term only
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
    # image-only (no substrate channels) counterparts: the substrate channels did not help (2 seeds), so the
    # architecture and adversarial-loss comparisons are repeated on the image-only reference 's2f_no_substrate'
    "unet_no_substrate":                 dict(arch="unet", lambda_gan=0.0, in_channels=1, substrate_mode="none"),
    "unet_gan_no_substrate":             dict(arch="unet", in_channels=1, substrate_mode="none"),
    "s2f_no_substrate_no_gan":           dict(in_channels=1, substrate_mode="none", lambda_gan=0.0),
    "s2f_no_substrate_no_multiscale":    dict(in_channels=1, substrate_mode="none", use_multi_scale_input=False),
    "s2f_no_substrate_bridge_plain":     dict(in_channels=1, substrate_mode="none", bridge="plain"),
    "s2f_no_substrate_no_attention_gates": dict(in_channels=1, substrate_mode="none", attention_gates=False),
    # staged substrate conditioning, as in the released model: phase 1 is the image-only phase 1 of
    # 's2f_no_substrate' (same seed, copied); the first convolution(s) are then extended to the two substrate
    # channels with zero-initialised weights (function-preserving), and phase 2 is trained with the substrate
    # channels. Control = 's2f_no_substrate' (same phase 1, phase 2 without substrate channels).
    "s2f_staged":                        dict(share_phase1_with="s2f_no_substrate", expand_from_image_only=True),
    "s2f_staged_no_force":               dict(share_phase1_with="s2f_no_substrate", expand_from_image_only=True,
                                              lambda_force=0.0),
}

# Recommended order if GPU time is limited (S2F about 7 h per run on an RTX A6000, U-Net about 2.5 h)
PRIORITY = ["s2f_full", "unet", "unet_gan", "s2f_no_gan", "s2f_no_substrate", "s2f_no_force",
            "s2f_no_attention_gates", "s2f_no_multiscale", "s2f_bridge_plain",
            "s2f_bridge_no_cbam", "s2f_nominal_substrate", "s2f_gan_0.1", "s2f_gan_5"]
# 'mask_only' needs a cell mask (cell_mask.tif) in every sample folder and is therefore not in PRIORITY.


def get_config(variant, epochs=None, g_lr=None, d_lr=None, **overrides):
    """epochs / g_lr / d_lr override every phase (smoke tests); other overrides are top-level keys."""
    if variant not in VARIANTS:
        raise KeyError(f"unknown variant '{variant}'. Options: {', '.join(VARIANTS)}")
    cfg = copy.deepcopy(PROTOCOL)
    cfg.update(VARIANTS[variant])
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    cfg["variant"] = variant
    if cfg["substrate_mode"] == "none":
        cfg["in_channels"] = 1
    for ph in cfg["phases"]:
        if ph["lambda_force"] == "from_variant":
            ph["lambda_force"] = cfg["lambda_force"]
        for k, v in (("epochs", epochs), ("g_lr", g_lr), ("d_lr", d_lr)):
            if v is not None:
                ph[k] = v
    cfg["epochs"] = sum(ph["epochs"] for ph in cfg["phases"])     # total, used to detect finished runs
    return cfg
