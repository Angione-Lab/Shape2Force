"""
Evaluate released checkpoints, baselines and trained ablation runs on every split manifest.

    python -m experiments.evaluate --released --meanmap               # released models + input-free baseline
    python -m experiments.evaluate --released --sweep                  # + substrate-setting sensitivity
    python -m experiments.evaluate --stage1 <path>/last_checkpoint_final.pth   # image-only stage-1 model
    python -m experiments.evaluate --runs                              # every experiments/runs/*/best_checkpoint.pth

Metrics are computed PER IMAGE (mean and SD are then taken across images):
    mse, ms_ssim, pixel_r (Pearson), rme (per-image WFM-RME), mean_gt / mean_pred (heatmap units),
    mean_gt_Pa / mean_pred_Pa (x colour-scale limit of the set, 500 Pa single cells, 50 Pa spheroids),
    gt_frac_saturated (pixels >= 0.98 of the colour limit), inference time.
Outputs (experiments/results/eval/ by default; existing rows of other models are kept):
    per_image.csv, summary.csv
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from experiments.ablation_models import build_generator, n_params
from experiments.layout import DEFAULT_DATASET_ROOT, RESULTS_DIR, RUNS_DIR, S2F_ROOT, list_manifests
from experiments.losses import PerImageWFMRMELoss
from experiments.manifest_data import make_loader
from models.s2f_model import create_s2f_model, create_settings_channels
from utils.metrics import TorchMetricsWrapper
from utils.substrate_settings import compute_settings_normalization

CONFIG_PATH = os.path.join(S2F_ROOT, "config", "substrate_settings.json")
SWEEP_KEYS = ["PDMS_1kPa_3um", "fibroblasts_PDMS", "PDMS_10kPa_3um"]


def released_models(device):
    out = {}
    g, _ = create_s2f_model(in_channels=3, model_type="s2f")
    g.load_state_dict(torch.load(os.path.join(S2F_ROOT, "ckp/single_cell/ckp_singlecell_GN.pth"), map_location="cpu",
                                 weights_only=False)["generator_state_dict"], strict=True)
    out["released_singlecell"] = dict(model=g.to(device).eval(), kind="singlecell", in_channels=3, tanh=True)
    g, _ = create_s2f_model(in_channels=1, model_type="s2f")
    g.load_state_dict(torch.load(os.path.join(S2F_ROOT, "ckp/spheroid/ckp_spheroid_GN.pth"), map_location="cpu",
                                 weights_only=False)["generator_state_dict"], strict=True)
    out["released_spheroid_GN"] = dict(model=g.to(device).eval(), kind="spheroid", in_channels=1, tanh=True)
    fn = os.path.join(S2F_ROOT, "ckp/spheroid/ckp_spheroid_FN.pth")
    if os.path.exists(fn):
        g, _ = create_s2f_model(in_channels=1, model_type="s2f_spheroid")
        g.load_state_dict(torch.load(fn, map_location="cpu", weights_only=False)["model_state_dict"], strict=True)
        g.set_output_mode(use_tanh=False)
        out["supervised_spheroid_FN"] = dict(model=g.to(device).eval(), kind="spheroid", in_channels=1, tanh=False)
    return out


def stage1_model(path, device):
    g, _ = create_s2f_model(in_channels=1, model_type="s2f")
    sd = torch.load(path, map_location="cpu", weights_only=False)["generator_state_dict"]
    g.load_state_dict({k.replace("denoise_module", "initial_conv"): v for k, v in sd.items()}, strict=True)
    return dict(model=g.to(device).eval(), kind="singlecell", in_channels=1, tanh=True)


def run_models(device, pattern):
    out = {}
    for p in sorted(glob.glob(pattern)):
        h = os.path.join(os.path.dirname(p), "history.json")
        ck = torch.load(p, map_location="cpu", weights_only=False)
        cfg = ck["config"]
        done = os.path.exists(h) and len(json.load(open(h))) >= cfg.get("epochs", 0)
        if not done and not os.environ.get("S2F_EVAL_UNFINISHED"):
            print(f"  skipping unfinished run {os.path.basename(os.path.dirname(p))} (set S2F_EVAL_UNFINISHED=1 to include)")
            continue
        g = build_generator(**cfg); g.load_state_dict(ck["generator_state_dict"], strict=True)
        name = os.path.basename(os.path.dirname(p))
        out[name] = dict(model=g.to(device).eval(), kind="singlecell", in_channels=cfg["in_channels"], tanh=True,
                         substrate_mode=cfg["substrate_mode"], input_mode=cfg["input_mode"],
                         epoch=ck.get("epoch"), val_loss=ck.get("val_loss"))
    return out


def mean_map(manifest, dataset_root):
    acc, n = None, 0
    for b in make_loader(manifest, dataset_root, batch_size=8, substrate_mode="none"):
        acc = b[1].sum(0, keepdim=True) if acc is None else acc + b[1].sum(0, keepdim=True); n += b[1].size(0)
    return acc / n


def evaluate_one(name, spec, manifest, dataset_root, device, mw, norm, substrate_mode, batch, mean_pred=None):
    input_mode = spec.get("input_mode", "bf")
    use_settings = spec.get("in_channels", 1) == 3
    sm = substrate_mode if use_settings else "none"
    forced = None
    if sm and sm.startswith("force:"):
        forced, sm = sm.split(":", 1)[1], "reference"
    try:
        loader = make_loader(manifest, dataset_root, batch_size=batch, substrate_mode=sm, input_mode=input_mode)
    except FileNotFoundError as e:
        print(f"  [skip] {name} on {manifest}: {e}"); return []
    rows_meta, rec, out, k = loader.dataset.rows, PerImageWFMRMELoss(), [], 0
    with torch.no_grad():
        for b in loader:
            x, y = b[0].to(device), b[1].to(device)
            t0 = time.time()
            if mean_pred is not None:
                p = mean_pred.to(device).expand_as(y)
            else:
                if use_settings:
                    meta = b[4]
                    if forced:
                        meta = dict(meta, substrate=[forced] * x.size(0))
                    x = torch.cat([x, create_settings_channels(meta, norm, device, x.shape, config_path=CONFIG_PATH)], 1)
                p = spec["model"](x)
                p = (p + 1) / 2 if spec.get("tanh", True) else p
            dt = (time.time() - t0) / x.size(0)
            for i in range(y.size(0)):
                r = rows_meta[k]; k += 1
                t, q = y[i:i + 1], p[i:i + 1]
                yt = float(r.get("ythreshold_Pa") or (50 if "spheroid" in manifest else 500))
                out.append({"model": name, "manifest": manifest, "substrate_mode": substrate_mode if use_settings else "n/a",
                            "sample_dir": r["sample_dir"], "set": r.get("set", ""), "condition": r.get("condition", ""),
                            "mse": mw.compute_mse(t, q), "ms_ssim": mw.compute_ms_ssim(t, q),
                            "pixel_r": float(np.corrcoef(t.flatten().cpu().numpy(), q.flatten().cpu().numpy())[0, 1]),
                            "rme": float(rec(q, t)), "mean_gt": float(t.mean()), "mean_pred": float(q.mean()),
                            "mean_gt_Pa": float(t.mean()) * yt, "mean_pred_Pa": float(q.mean()) * yt,
                            "gt_max": float(t.max()), "gt_frac_saturated": float((t >= 0.98).float().mean()),
                            "seconds_per_image": dt})
    return out


def summarise(df):
    df = df.assign(substrate_mode=df["substrate_mode"].fillna("none"))   # "n/a" is read back from CSV as NaN
    g = df.groupby(["model", "manifest", "substrate_mode"], dropna=False)
    s = g.agg(n=("mse", "size"), **{f"{m}_{f}": (m, f) for m in ["mse", "ms_ssim", "pixel_r", "rme", "mean_gt_Pa",
                                                                   "mean_pred_Pa", "gt_frac_saturated"] for f in ["mean", "std"]},
              seconds_per_image=("seconds_per_image", "mean")).reset_index()
    s["pred_over_gt"] = s["mean_pred_Pa_mean"] / s["mean_gt_Pa_mean"]
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--released", action="store_true", help="released single-cell + spheroid checkpoints")
    ap.add_argument("--sweep", action="store_true", help="released single-cell model with 1 / 6 / 10 kPa settings")
    ap.add_argument("--meanmap", action="store_true", help="input-free baseline (mean training map)")
    ap.add_argument("--stage1", default=None, help="path to last_checkpoint_final.pth (image-only stage 1)")
    ap.add_argument("--runs", nargs="?", const=os.path.join(RUNS_DIR, "*", "best_checkpoint.pth"), default=None,
                    help="glob of trained run checkpoints (default experiments/runs/*/best_checkpoint.pth)")
    ap.add_argument("--manifests", default="", help="comma-separated manifest names (default: all in experiments/splits)")
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "eval"))
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--device", default=None)
    ap.add_argument("--run-modes", default=None,
                    help="substrate modes for trained runs with substrate channels, e.g. legacy,nominal (default: run config)")
    a = ap.parse_args()
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    mans = [m for m in (a.manifests.split(",") if a.manifests else list_manifests()) if m]
    sc_mans = [m for m in mans if not m.startswith("spheroid") and m != "alltune_train" and m != "zenodo_v1_train"]
    sph_mans = [m for m in mans if m.startswith("spheroid") and m != "spheroid_train"]
    print(f"device {device}; single-cell manifests {sc_mans}; spheroid manifests {sph_mans}")
    mw, norm = TorchMetricsWrapper(device=device), compute_settings_normalization(config_path=CONFIG_PATH)

    models = {}
    if a.released:
        models.update(released_models(device))
    if a.stage1:
        models["stage1_image_only"] = stage1_model(a.stage1, device)
    if a.runs:
        models.update(run_models(device, a.runs))
    rows, params = [], []
    for name, spec in models.items():
        params.append({"model": name, "n_params_M": n_params(spec["model"]) / 1e6, "epoch": spec.get("epoch"),
                       "val_loss": spec.get("val_loss")})
        targets = sc_mans if spec["kind"] == "singlecell" else sph_mans
        modes = [spec.get("substrate_mode", "legacy")]
        if a.run_modes and spec.get("in_channels", 1) == 3 and name not in ("released_singlecell",):
            modes = [m for m in a.run_modes.split(",") if m]
        if name == "released_singlecell":
            modes = ["legacy", "nominal"] + ([f"force:{k}" for k in SWEEP_KEYS] if a.sweep else [])
        for m in targets:
            for sm in modes:
                print(f"  {name:28s} {m:28s} {sm}", flush=True)
                rows += evaluate_one(name, spec, m, a.dataset_root, device, mw, norm, sm, a.batch)
    if a.meanmap:
        for train_m, targets, name in [("alltune_train", sc_mans, "baseline_mean_map_singlecell"),
                                       ("spheroid_train", sph_mans, "baseline_mean_map_spheroid")]:
            if train_m not in list_manifests():
                continue
            M = mean_map(train_m, a.dataset_root)
            for m in targets:
                print(f"  {name:28s} {m}", flush=True)
                rows += evaluate_one(name, {"in_channels": 1}, m, a.dataset_root, device, mw, norm, "n/a", a.batch, mean_pred=M)
    os.makedirs(a.out, exist_ok=True)
    new = pd.DataFrame(rows)
    pi = os.path.join(a.out, "per_image.csv")
    if os.path.exists(pi) and len(new):           # keep rows of models not re-evaluated now
        old = pd.read_csv(pi)
        new = pd.concat([old[~old.model.isin(new.model.unique())], new], ignore_index=True)
    if len(new):
        new.to_csv(pi, index=False)
        summarise(new).to_csv(os.path.join(a.out, "summary.csv"), index=False)
    pp = os.path.join(a.out, "model_params.csv")
    if params:
        P = pd.DataFrame(params)
        if os.path.exists(pp):
            old = pd.read_csv(pp); P = pd.concat([old[~old.model.isin(P.model)], P], ignore_index=True)
        P.to_csv(pp, index=False)
    print(f"wrote {pi}, summary.csv, model_params.csv in {a.out}")


if __name__ == "__main__":
    main()
