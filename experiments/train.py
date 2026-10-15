"""
Train one baseline / ablation variant with the shared protocol (experiments/configs.py).

    python -m experiments.train --variant s2f_full
    python -m experiments.train --variant unet --seed 1
    python -m experiments.train --variant s2f_no_gan --resume          # continue from last_checkpoint.pth
    python -m experiments.train --variant unet --epochs 1 --max-batches 5   # smoke test

Outputs in experiments/runs/<variant>_seed<seed>/:
    config.json            exact configuration used
    history.json           per-epoch losses (each generator loss term separately), lr, time
    best_checkpoint.pth    generator weights at the lowest validation reconstruction loss
    last_checkpoint.pth    full state for --resume
    visualizations/        predictions on the first validation batch every 10 epochs
The training loop follows training/s2f_trainer.py (same losses, label convention and update order);
it additionally skips the discriminator when lambda_gan = 0, records every loss term, and seeds all RNGs.
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.ablation_models import build_discriminator, build_generator, n_params
from experiments.configs import get_config
from experiments.layout import DEFAULT_DATASET_ROOT, RUNS_DIR, S2F_ROOT
from experiments.losses import PerImageWFMRMELoss
from experiments.manifest_data import make_loader
from models.s2f_model import create_settings_channels
from utils.substrate_settings import compute_settings_normalization

CONFIG_PATH = os.path.join(S2F_ROOT, "config", "substrate_settings.json")


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def cosine_with_warmup(optimizer, warmup_steps, total_steps):
    """Same schedule as diffusers.optimization.get_cosine_schedule_with_warmup (num_cycles=0.5)."""
    def f(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def add_settings(x, batch, cfg, norm, device):
    if cfg["substrate_mode"] == "none":
        return x
    meta = batch[4]
    return torch.cat([x, create_settings_channels(meta, norm, device, x.shape, config_path=CONFIG_PATH)], dim=1)


def save_vis(G, val_loader, cfg, norm, device, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    batch = next(iter(val_loader))
    x = add_settings(batch[0].to(device), batch, cfg, norm, device)
    with torch.no_grad():
        p = (G(x) + 1) / 2
    n = min(2, x.size(0))
    fig, ax = plt.subplots(3, n, figsize=(4 * n, 12), squeeze=False)
    for i in range(n):
        ax[0, i].imshow(x[i, 0].cpu(), cmap="gray"); ax[1, i].imshow(p[i, 0].cpu(), cmap="jet", vmin=0, vmax=1)
        ax[2, i].imshow(batch[1][i, 0], cmap="jet", vmin=0, vmax=1)
        for r in range(3):
            ax[r, i].axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def train(cfg, run_dir, dataset_root=DEFAULT_DATASET_ROOT, device=None, resume=False, max_batches=None,
          num_workers=2, allow_cpu=False):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if str(device).startswith("cpu") and not allow_cpu:
        raise SystemExit("CUDA is not available - refusing to train on CPU (pass --allow-cpu for a smoke test)")
    if str(device).startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(torch.device(device))}  "
              f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')})")
    os.makedirs(os.path.join(run_dir, "visualizations"), exist_ok=True)
    seed_everything(cfg["seed"])
    train_loader = make_loader(cfg["train_manifest"], dataset_root, cfg["batch_size"], cfg["img_size"],
                               augment=cfg["augment"], shuffle=True, substrate_mode=cfg["substrate_mode"],
                               input_mode=cfg["input_mode"], seed=cfg["seed"], num_workers=num_workers)
    if num_workers > 0:                      # check that worker processes work on this machine
        try:
            next(iter(train_loader))
        except RuntimeError as e:
            print(f"DataLoader workers failed ({str(e).splitlines()[0][:120]}); falling back to num_workers=0")
            train_loader = make_loader(cfg["train_manifest"], dataset_root, cfg["batch_size"], cfg["img_size"],
                                       augment=cfg["augment"], shuffle=True, substrate_mode=cfg["substrate_mode"],
                                       input_mode=cfg["input_mode"], seed=cfg["seed"], num_workers=0)
        seed_everything(cfg["seed"])
    val_loader = make_loader(cfg["val_manifest"], dataset_root, cfg["batch_size"], cfg["img_size"],
                             substrate_mode=cfg["substrate_mode"], input_mode=cfg["input_mode"])
    norm = compute_settings_normalization(config_path=CONFIG_PATH)
    use_gan = cfg["lambda_gan"] > 0
    G = build_generator(**cfg).to(device)
    D = build_discriminator(cfg["in_channels"]).to(device) if use_gan else None
    cfg = dict(cfg, n_params_generator=n_params(G), n_params_discriminator=n_params(D) if D else 0,
               n_train=len(train_loader.dataset), n_val=len(val_loader.dataset), device=str(device),
               torch_version=torch.__version__)
    json.dump(cfg, open(os.path.join(run_dir, "config.json"), "w"), indent=2)
    print(f"[{cfg['variant']}] G {cfg['n_params_generator']/1e6:.2f} M params, D {cfg['n_params_discriminator']/1e6:.2f} M; "
          f"train {cfg['n_train']}, val {cfg['n_val']}, device {device}")

    g_opt = torch.optim.Adam(G.parameters(), lr=cfg["g_lr"], betas=(cfg["beta1"], cfg["beta2"]))
    d_opt = torch.optim.Adam(D.parameters(), lr=cfg["d_lr"], betas=(cfg["beta1"], cfg["beta2"])) if use_gan else None
    steps = len(train_loader) * cfg["epochs"]
    g_sch = cosine_with_warmup(g_opt, int(steps * cfg["warmup_frac"]), steps)
    d_sch = cosine_with_warmup(d_opt, int(steps * cfg["warmup_frac"]), steps) if use_gan else None
    rec_loss, bce = PerImageWFMRMELoss(), nn.BCEWithLogitsLoss()
    hist, best, start = [], float("inf"), 0

    last_path = os.path.join(run_dir, "last_checkpoint.pth")
    if resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        G.load_state_dict(ck["generator_state_dict"]); g_opt.load_state_dict(ck["g_optimizer"]); g_sch.load_state_dict(ck["g_scheduler"])
        if use_gan:
            D.load_state_dict(ck["discriminator_state_dict"]); d_opt.load_state_dict(ck["d_optimizer"]); d_sch.load_state_dict(ck["d_scheduler"])
        hist, best, start = ck["history"], ck["best_val_loss"], ck["epoch"] + 1
        print(f"resumed at epoch {start}")

    for epoch in range(start, cfg["epochs"]):
        t0 = time.time(); G.train(); D.train() if use_gan else None
        acc = dict(g=0.0, rec=0.0, gan=0.0, force=0.0, d=0.0); nb = 0
        for bi, batch in enumerate(train_loader):
            if max_batches and bi >= max_batches:
                break
            x = add_settings(batch[0].to(device), batch, cfg, norm, device)
            y = batch[1].to(device)
            fake = G(x); p = (fake + 1) / 2
            l_rec = rec_loss(p, y)
            l_force = F.mse_loss(p.flatten(1).mean(1), y.flatten(1).mean(1))
            l_gan = torch.zeros((), device=device)
            if use_gan:
                pf = D(torch.cat([x, fake], 1)); l_gan = bce(pf, torch.ones_like(pf))
            l_g = cfg["lambda_rec"] * l_rec + cfg["lambda_gan"] * l_gan + cfg["lambda_force"] * l_force
            g_opt.zero_grad(); l_g.backward(); g_opt.step()
            l_d = torch.zeros(())
            if use_gan:
                pr = D(torch.cat([x, y * 2 - 1], 1)); pf = D(torch.cat([x, fake.detach()], 1))
                l_d = 0.5 * (bce(pr, torch.ones_like(pr)) + bce(pf, torch.zeros_like(pf)))
                d_opt.zero_grad(); l_d.backward(); d_opt.step(); d_sch.step()
            g_sch.step()
            for k, v in zip(acc, [l_g, l_rec, l_gan, l_force, l_d]):
                acc[k] += float(v)
            nb += 1
        G.eval(); v_rec = v_mse = 0.0; nv = 0
        with torch.no_grad():
            for bi, batch in enumerate(val_loader):
                if max_batches and bi >= max_batches:
                    break
                x = add_settings(batch[0].to(device), batch, cfg, norm, device); y = batch[1].to(device)
                p = (G(x) + 1) / 2
                v_rec += float(rec_loss(p, y)); v_mse += float(F.mse_loss(p, y)); nv += 1
        h = {"epoch": epoch, **{f"train_{k}": v / max(nb, 1) for k, v in acc.items()},
             "val_rec": v_rec / max(nv, 1), "val_mse": v_mse / max(nv, 1),
             "g_lr": g_opt.param_groups[0]["lr"], "epoch_seconds": time.time() - t0}
        hist.append(h)
        is_best = h["val_rec"] < best
        best = min(best, h["val_rec"])
        print(f"epoch {epoch:3d}  G {h['train_g']:.4f} (rec {h['train_rec']:.4f} gan {h['train_gan']:.4f} "
              f"force {h['train_force']:.5f})  D {h['train_d']:.4f}  val_rec {h['val_rec']:.4f}  "
              f"{'*' if is_best else ' '}  {h['epoch_seconds']:.0f}s", flush=True)
        json.dump(hist, open(os.path.join(run_dir, "history.json"), "w"), indent=1)
        if is_best:
            torch.save({"epoch": epoch, "generator_state_dict": G.state_dict(), "config": cfg,
                        "val_loss": h["val_rec"], "history": hist}, os.path.join(run_dir, "best_checkpoint.pth"))
        state = {"epoch": epoch, "generator_state_dict": G.state_dict(), "g_optimizer": g_opt.state_dict(),
                 "g_scheduler": g_sch.state_dict(), "history": hist, "best_val_loss": best, "config": cfg}
        if use_gan:
            state.update(discriminator_state_dict=D.state_dict(), d_optimizer=d_opt.state_dict(), d_scheduler=d_sch.state_dict())
        torch.save(state, last_path)
        if epoch % 10 == 0 or epoch == cfg["epochs"] - 1:
            save_vis(G, val_loader, cfg, norm, device, os.path.join(run_dir, "visualizations", f"epoch_{epoch:03d}.png"))
    print(f"done. best val_rec {best:.4f}; checkpoints in {run_dir}")
    return run_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--g-lr", type=float, default=None); ap.add_argument("--d-lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-batches", type=int, default=None, help="limit batches per epoch (smoke test only)")
    ap.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes for augmentation")
    ap.add_argument("--allow-cpu", action="store_true", help="allow training without CUDA (smoke tests)")
    a = ap.parse_args()
    cfg = get_config(a.variant, seed=a.seed, epochs=a.epochs, g_lr=a.g_lr, d_lr=a.d_lr, batch_size=a.batch_size)
    run_dir = a.run_dir or os.path.join(RUNS_DIR, f"{a.variant}_seed{cfg['seed']}")
    train(cfg, run_dir, a.dataset_root, a.device, a.resume, a.max_batches, a.num_workers, a.allow_cpu)


if __name__ == "__main__":
    main()
