"""
Train one baseline / ablation variant with the shared protocol (experiments/configs.py).

    python -m experiments.train --variant s2f_full
    python -m experiments.train --variant unet --seed 1
    python -m experiments.train --variant s2f_no_gan --resume          # continue an interrupted run
    python -m experiments.train --variant unet --epochs 1 --max-batches 5   # smoke test (1 epoch per phase)

Two phases (see experiments/configs.py). Outputs in experiments/runs/<variant>_seed<seed>/:
    config.json            exact configuration used
    phase1/, phase2/       per phase: history.json (every loss term, val loss / WFM-RME / pixel r, lr, time),
                           best_checkpoint.pth (lowest validation loss of that phase), last_checkpoint.pth
                           (full state for --resume), visualizations/ (first validation batch every 10 epochs)
    best_checkpoint.pth    final model = best phase-2 generator (written when the run is complete)
    history.json           both phases concatenated (written when the run is complete)
The training loop follows training/s2f_trainer.py (same losses, label convention and update order);
it additionally skips the discriminator when lambda_gan = 0, records every loss term, and seeds all RNGs.
"""
import argparse
import json
import math
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.ablation_models import build_discriminator, build_generator, n_params
from experiments.configs import get_config
from experiments.layout import DEFAULT_DATASET_ROOT, RUNS_DIR, S2F_ROOT
from experiments.losses import REC_LOSSES, PerImageWFMRMELoss
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


def pixel_r(p, y):
    """Mean per-image Pearson correlation between predicted and true maps (monitoring only)."""
    p, y = p.flatten(1), y.flatten(1)
    p, y = p - p.mean(1, keepdim=True), y - y.mean(1, keepdim=True)
    return ((p * y).sum(1) / (p.norm(dim=1) * y.norm(dim=1) + 1e-12)).mean()


def _done(phase_dir, epochs):
    h = os.path.join(phase_dir, "history.json")
    return (os.path.exists(h) and len(json.load(open(h))) >= epochs
            and os.path.exists(os.path.join(phase_dir, "best_checkpoint.pth")))


def load_expanding(G, state):
    """Load a state dict into G; input convolutions saved with 1 input channel are extended to G's input channels
    with zero weights for the new channels (function-preserving expansion, as in the released model)."""
    own, new, expanded = G.state_dict(), {}, []
    for k, v in state.items():
        t = own.get(k)
        if t is not None and v.dim() == 4 and t.dim() == 4 and v.shape[1] == 1 and t.shape[1] > 1 \
                and v.shape[0] == t.shape[0] and v.shape[2:] == t.shape[2:]:
            w = torch.zeros_like(t); w[:, :1] = v.to(t.dtype); new[k] = w; expanded.append(k)
        else:
            new[k] = v
    G.load_state_dict(new, strict=True)
    return expanded


def run_phase(G, ph, phase_dir, cfg, train_loader, val_loader, norm, device, resume, max_batches):
    """One training phase. Fresh discriminator, optimisers and schedule (as in the original staged training);
    the generator continues from its current weights. Checkpoint selection: lowest validation loss of this phase."""
    os.makedirs(os.path.join(phase_dir, "visualizations"), exist_ok=True)
    use_gan = cfg["lambda_gan"] > 0
    D = build_discriminator(cfg["in_channels"]).to(device) if use_gan else None
    g_opt = torch.optim.Adam(G.parameters(), lr=ph["g_lr"], betas=(cfg["beta1"], cfg["beta2"]))
    d_opt = torch.optim.Adam(D.parameters(), lr=ph["d_lr"], betas=(cfg["beta1"], cfg["beta2"])) if use_gan else None
    n_batches = min(len(train_loader), max_batches) if max_batches else len(train_loader)
    steps = n_batches * ph["epochs"]
    g_sch = cosine_with_warmup(g_opt, int(steps * cfg["warmup_frac"]), steps)
    d_sch = cosine_with_warmup(d_opt, int(steps * cfg["warmup_frac"]), steps) if use_gan else None
    rec_loss, rme, bce = REC_LOSSES[ph["rec_loss"]](), PerImageWFMRMELoss(), nn.BCEWithLogitsLoss()
    hist, best, start = [], float("inf"), 0
    last_path = os.path.join(phase_dir, "last_checkpoint.pth")
    if resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        G.load_state_dict(ck["generator_state_dict"]); g_opt.load_state_dict(ck["g_optimizer"]); g_sch.load_state_dict(ck["g_scheduler"])
        if use_gan:
            D.load_state_dict(ck["discriminator_state_dict"]); d_opt.load_state_dict(ck["d_optimizer"]); d_sch.load_state_dict(ck["d_scheduler"])
        hist, best, start = ck["history"], ck["best_val_loss"], ck["epoch"] + 1
        print(f"[{ph['name']}] resumed at epoch {start}")
    print(f"[{ph['name']}] {ph['epochs']} epochs, loss {ph['rec_loss']} x{ph['lambda_rec']:g}, "
          f"GAN x{cfg['lambda_gan']:g}, force x{ph['lambda_force']:g}, lr {ph['g_lr']:g}", flush=True)

    for epoch in range(start, ph["epochs"]):
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
            l_g = ph["lambda_rec"] * l_rec + cfg["lambda_gan"] * l_gan + ph["lambda_force"] * l_force
            g_opt.zero_grad(); l_g.backward(); g_opt.step()
            l_d = torch.zeros(())
            if use_gan:
                pr = D(torch.cat([x, y * 2 - 1], 1)); pf = D(torch.cat([x, fake.detach()], 1))
                l_d = 0.5 * (bce(pr, torch.ones_like(pr)) + bce(pf, torch.zeros_like(pf)))
                d_opt.zero_grad(); l_d.backward(); d_opt.step(); d_sch.step()
            g_sch.step()
            for k, v in zip(acc, [l_g, l_rec, l_gan, l_force, l_d]):
                acc[k] += float(v.detach())
            nb += 1
        G.eval(); v = dict(rec=0.0, rme=0.0, mse=0.0, r=0.0); nv = 0
        with torch.no_grad():
            for bi, batch in enumerate(val_loader):
                if max_batches and bi >= max_batches:
                    break
                x = add_settings(batch[0].to(device), batch, cfg, norm, device); y = batch[1].to(device)
                p = (G(x) + 1) / 2
                v["rec"] += float(rec_loss(p, y)); v["rme"] += float(rme(p, y))
                v["mse"] += float(F.mse_loss(p, y)); v["r"] += float(pixel_r(p, y)); nv += 1
        h = {"phase": ph["name"], "epoch": epoch, **{f"train_{k}": a / max(nb, 1) for k, a in acc.items()},
             **{f"val_{k}": a / max(nv, 1) for k, a in v.items()},
             "g_lr": g_opt.param_groups[0]["lr"], "epoch_seconds": time.time() - t0}
        hist.append(h)
        is_best = h["val_rec"] < best
        best = min(best, h["val_rec"])
        print(f"[{ph['name']}] epoch {epoch:3d}  G {h['train_g']:.4f} (rec {h['train_rec']:.4f} gan {h['train_gan']:.4f} "
              f"force {h['train_force']:.5f})  D {h['train_d']:.4f}  val_rec {h['val_rec']:.4f}  "
              f"val_r {h['val_r']:.3f}  val_rme {h['val_rme']:.3f}  {'*' if is_best else ' '}  {h['epoch_seconds']:.0f}s", flush=True)
        if is_best:
            torch.save({"epoch": epoch, "phase": ph["name"], "generator_state_dict": G.state_dict(), "config": cfg,
                        "val_loss": h["val_rec"], "history": hist}, os.path.join(phase_dir, "best_checkpoint.pth"))
        state = {"epoch": epoch, "generator_state_dict": G.state_dict(), "g_optimizer": g_opt.state_dict(),
                 "g_scheduler": g_sch.state_dict(), "history": hist, "best_val_loss": best, "config": cfg}
        if use_gan:
            state.update(discriminator_state_dict=D.state_dict(), d_optimizer=d_opt.state_dict(), d_scheduler=d_sch.state_dict())
        torch.save(state, last_path)
        json.dump(hist, open(os.path.join(phase_dir, "history.json"), "w"), indent=1)
        if epoch % 10 == 0 or epoch == ph["epochs"] - 1:
            save_vis(G, val_loader, cfg, norm, device, os.path.join(phase_dir, "visualizations", f"epoch_{epoch:03d}.png"))
    return hist


def train(cfg, run_dir, dataset_root=DEFAULT_DATASET_ROOT, device=None, resume=False, max_batches=None,
          num_workers=2, allow_cpu=False):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if str(device).startswith("cpu") and not allow_cpu:
        raise SystemExit("CUDA is not available - refusing to train on CPU (pass --allow-cpu for a smoke test)")
    if str(device).startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(torch.device(device))}  "
              f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')})")
    os.makedirs(run_dir, exist_ok=True)
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
    G = build_generator(**cfg).to(device)
    n_d = n_params(build_discriminator(cfg["in_channels"])) if cfg["lambda_gan"] > 0 else 0
    cfg = dict(cfg, n_params_generator=n_params(G), n_params_discriminator=n_d,
               n_train=len(train_loader.dataset), n_val=len(val_loader.dataset), device=str(device),
               torch_version=torch.__version__)
    json.dump(cfg, open(os.path.join(run_dir, "config.json"), "w"), indent=2)
    print(f"[{cfg['variant']}] G {cfg['n_params_generator']/1e6:.2f} M params, D {n_d/1e6:.2f} M; "
          f"train {cfg['n_train']}, val {cfg['n_val']}, device {device}", flush=True)

    all_hist = []
    for ph in cfg["phases"]:
        phase_dir = os.path.join(run_dir, ph["name"])
        shared = cfg.get("share_phase1_with")
        if ph["name"] == "phase1" and shared and not _done(phase_dir, ph["epochs"]):
            src = os.path.join(os.path.dirname(run_dir), f"{shared}_seed{cfg['seed']}", "phase1")
            if _done(src, ph["epochs"]):
                print(f"[phase1] identical to {shared} (same seed, data and phase-1 settings) - copying {src}")
                shutil.copytree(src, phase_dir, dirs_exist_ok=True)
            elif cfg.get("expand_from_image_only"):
                raise SystemExit(f"{cfg['variant']} needs the finished image-only phase 1 of {shared} seed "
                                 f"{cfg['seed']} in {src}")
            else:
                print(f"[phase1] {shared} phase 1 not finished yet - training phase 1 for this run")
        if _done(phase_dir, ph["epochs"]) and (resume or shared):
            print(f"[{ph['name']}] already finished - skipping")
            hist = json.load(open(os.path.join(phase_dir, "history.json")))
        else:
            hist = run_phase(G, ph, phase_dir, cfg, train_loader, val_loader, norm, device, resume, max_batches)
        all_hist += hist
        best_ck = torch.load(os.path.join(phase_dir, "best_checkpoint.pth"), map_location="cpu", weights_only=False)
        if cfg.get("expand_from_image_only") and ph["name"] == "phase1":
            ex = load_expanding(G, best_ck["generator_state_dict"])   # image-only phase 1 -> + substrate channels
            print(f"[phase1] image-only generator loaded; zero-initialised substrate weights added to {ex}")
        else:
            G.load_state_dict(best_ck["generator_state_dict"])      # next phase starts from this phase's best
    best_ck["config"] = cfg
    torch.save(best_ck, os.path.join(run_dir, "best_checkpoint.pth"))   # final model = best of the last phase
    json.dump(all_hist, open(os.path.join(run_dir, "history.json"), "w"), indent=1)   # written last = run finished
    print(f"done. final model: {best_ck['phase']} epoch {best_ck['epoch']}, val loss {best_ck['val_loss']:.4f}; in {run_dir}")
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
