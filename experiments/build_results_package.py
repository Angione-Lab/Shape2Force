"""Build the shared S2F results package (data, splits, predictions, per-image metrics, checkpoints).

    python -m experiments.build_results_package --stage1 <path>/last_checkpoint_final.pth --out <folder>
        [--dataset-root ...] [--skip-dataset] [--skip-png] [--with-staged] [--limit N] [--device cuda]

Layout written to --out (default: S2F_results_v2 next to the repository):

  README.md, CHECKSUMS.sha256
  splits/                      split CSVs used for training and evaluation (+ sets.csv = Table S2)
  dataset/single_cell/{train,validation,test}/<set>/<sample>/   bright field + GT map, as used by the model
  dataset/spheroid/{train,validation}/<id>/
  predictions/<model>/{train,validation,test}/<set>/<sample>/
      panel.png                bright field | GT | prediction, 0-500 Pa (spheroids 0-50 Pa), all metrics printed
      prediction.png           predicted traction-stress map with Pa colour bar
      prediction_Pa.tif        only with --with-tif: 32-bit float map in Pa (1024 x 1024, model resolution)
    models: released (single cell, BF + substrate channels, 6 kPa reference setting), image_only (stage i of the
            released model), spheroid_GN (paper), spheroid_FN (web-app default)
  results.xlsx (and results_csv/*.csv)
      README, sets, models (pooled + per set), tests (paired comparisons), test_images, validation_images,
      training_images (marked: not for evaluation), all_models (every evaluated model, from results/eval)
  checkpoints/                 released checkpoints + image-only + MODEL_CARD.md (+ retrained staged with --with-staged)

All numbers in results.xlsx come from this run (released / image-only / spheroid models) or from
experiments/results/eval/per_image.csv (all other models). Position_08 of DMSO SET II and Y-27632 SET II is the same
image; it is flagged (column `note`) and counted once in pooled test numbers.
"""
import argparse
import datetime
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

S2F_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if S2F_ROOT not in sys.path:
    sys.path.insert(0, S2F_ROOT)

from experiments.layout import DEFAULT_DATASET_ROOT, RESULTS_DIR, RUNS_DIR, SPLITS_DIR, manifest_path, read_manifest  # noqa: E402

SC_MANIFESTS = {"alltune_train": "train", "alltune_val": "validation", "test_fibro_1kPa_setIII": "test",
                "test_fibro_DMSO_setI": "test", "test_fibro_DMSO_setII": "test", "test_fibro_Y27632_setI": "test",
                "test_fibro_Y27632_setII": "test", "test_fibro_TGFb_setI": "test", "test_fibro_fibronectin_setI": "test",
                "test_u2os_6kPa": "test"}
SPH_MANIFESTS = {"spheroid_train": "train", "spheroid_val": "validation"}
DUP = {"singlecell_external/fibro_DMSO_setII/Position_08", "singlecell_external/fibro_Y27632_setII/Position_08"}
DUP_NOTE = "same image as Position_08 of the other drug set (DMSO SET II / Y-27632 SET II); correct label pending"
MODEL_INFO = {
    "released": ("Released S2F, single cell", "ckp/single_cell/ckp_singlecell_GN.pth",
                 "bright field + 2 substrate channels (6 kPa reference setting for every image)"),
    "image_only": ("Image-only S2F (stage i of the released model)", None, "bright field only"),
    "spheroid_GN": ("Released S2F, spheroid (paper model, Fig. 3G-I)", "ckp/spheroid/ckp_spheroid_GN.pth", "bright field"),
    "spheroid_FN": ("Supervised spheroid model (web-app default)", "ckp/spheroid/ckp_spheroid_FN.pth", "bright field"),
}


# --------------------------------------------------------------------------------------------- names
def set_label(r):
    sd, s = r["sample_dir"], r.get("set", "")
    if sd.startswith("singlecell/"):
        for tag in ["SET_VI", "SET_IV", "SET_V", "SET_III", "SET_II", "SET_I"]:
            if f"_{tag}_" in sd:
                return f"6kPa_{tag}"
        return "6kPa"
    lab = {"1kpa_new/train": "1kPa_train", "10kpa_new/train": "10kPa_train", "1kpa_new/test": "1kPa_SET_I&II",
           "10kpa_new/test": "10kPa_SET_I", "U2OS/test": "U2OS_6kPa", "spheroid/train": "spheroid",
           "spheroid/test": "spheroid"}
    return lab.get(s, s.replace("/", "_"))


def sample_label(r):
    b = os.path.basename(r["sample_dir"])
    if r["sample_dir"].startswith("singlecell/"):
        b = b.split("_Fibros_")[0]
    return b.replace(" ", "_").replace("(", "").replace(")", "")


def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------- figures
def save_maps(pdir, bf, gt_pa, pr_pa, vmax, d, title, plt, Image, with_tif=False):
    os.makedirs(pdir, exist_ok=True)
    if with_tif:
        Image.fromarray(pr_pa.astype(np.float32)).save(os.path.join(pdir, "prediction_Pa.tif"), compression="tiff_adobe_deflate")
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(pr_pa, cmap="jet", vmin=0, vmax=vmax); ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02); cb.set_label("Traction stress (Pa)")
    ax.set_title(f"Predicted traction stress ({d['model']})", fontsize=9)
    fig.savefig(os.path.join(pdir, "prediction.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.9), gridspec_kw=dict(wspace=0.04))
    lo, hi = np.percentile(bf, [1, 99])
    axs[0].imshow(bf, cmap="gray", vmin=lo, vmax=hi); axs[0].set_title("Bright field (contrast-stretched)", fontsize=9)
    axs[1].imshow(gt_pa, cmap="jet", vmin=0, vmax=vmax)
    axs[1].set_title(f"Ground truth (TFM)\nmean in force region {d['gt_region_mean_Pa']:.0f} Pa", fontsize=9)
    im = axs[2].imshow(pr_pa, cmap="jet", vmin=0, vmax=vmax)
    axs[2].set_title(f"Prediction ({d['model']})\nmean in force region {d['pred_region_mean_Pa']:.0f} Pa", fontsize=9)
    for a in axs:
        a.set_xticks([]); a.set_yticks([])
    cb = fig.colorbar(im, ax=axs[1:], fraction=0.023, pad=0.01); cb.set_label("Traction stress (Pa)", fontsize=8)
    txt = (f"MSE {d['mse']:.4f} | pixel r {d['pixel_r']:.3f} | RME {d['rme']:.3f} | MS-SSIM {d['ms_ssim']:.3f} | "
           f"pred/GT (field) {d['pred_gt_field']:.2f} | pred/GT (force region) {d['pred_gt_region']:.2f}")
    fig.text(0.5, 0.02, txt, ha="center", fontsize=8)
    fig.suptitle(title, fontsize=9, x=0.01, ha="left")
    fig.savefig(os.path.join(pdir, "panel.png"), dpi=110, bbox_inches="tight"); plt.close(fig)


# -------------------------------------------------------------------------------------------- tables
def pools(df):
    """Held-out pools used in the manuscript (Position_08 counted once / left out of the drug groups)."""
    s, dup_y = df["set_label"], df.sample_dir == "singlecell_external/fibro_Y27632_setII/Position_08"
    dup_d = df.sample_dir == "singlecell_external/fibro_DMSO_setII/Position_08"
    val, test = df.role == "validation", (df.role == "test") & ~s.isin(["U2OS_6kPa"])
    return {
        "Validation pool": val & df.kind.eq("single_cell"),
        "Test pool": test & ~dup_y & df.kind.eq("single_cell"),
        "1 kPa pool": s.isin(["1kPa_SET_I&II", "1kPa_SET_III"]) & df.role.ne("train"),
        "6 kPa pool": (s.eq("6kPa_SET_VI") & val) | (test & ~dup_y & ~s.eq("1kPa_SET_III")),
        "10 kPa pool": s.eq("10kPa_SET_I") & val,
        "DMSO": s.isin(["Control_SET_I", "DMSO_SET_II"]) & ~dup_d,
        "Y-27632": s.isin(["Y27632_SET_I", "Y27632_SET_II"]) & ~dup_y,
        "TGF-beta": s.eq("TGFB_SET_I"),
        "Spheroid validation": df.kind.eq("spheroid") & val,
    }


def summarise(d):
    return dict(n=len(d), MSE=d.mse.mean(), RME=d.rme.mean(), pixel_r=d.pixel_r.mean(), MS_SSIM=d.ms_ssim.mean(),
                pred_GT_field=d.pred_field_mean_Pa.sum() / max(d.gt_field_mean_Pa.sum(), 1e-12),
                pred_GT_region_geomean=float(np.exp(np.log(d.pred_region_mean_Pa / d.gt_region_mean_Pa).mean())))


def models_table(P):
    rows = []
    for m, g in P.groupby("model"):
        for name, mask in pools(g).items():
            sub = g[mask.loc[g.index]]
            if len(sub):
                rows.append(dict(model=m, set=name, kind="pool", **summarise(sub)))
        for (role, s), sub in g.groupby(["role", "set_label"]):
            if role != "train":
                rows.append(dict(model=m, set=f"{role}: {s}", kind="set", **summarise(sub)))
        tr = g[g.role == "train"]
        if len(tr):
            rows.append(dict(model=m, set="training images (not for evaluation)", kind="train", **summarise(tr)))
    return pd.DataFrame(rows)


def tests_table(eval_csv):
    """Paired Wilcoxon tests behind Tables S3a/S3b (test pool, 145 images; 6 kPa reference for substrate models)."""
    from scipy import stats
    if not os.path.exists(eval_csv):
        return pd.DataFrame()
    E = pd.read_csv(eval_csv); E["substrate_mode"] = E.substrate_mode.fillna("n/a")
    out = []
    for split, mask in [("validation (81)", E.manifest.eq("alltune_val")),
                        ("test (145)", E.manifest.str.startswith("test_fibro")
                         & E.sample_dir.ne("singlecell_external/fibro_Y27632_setII/Position_08"))]:
        S = E[mask]

        def M(models, mode=None):
            d = S[S.model.isin(models)]
            d = d[d.substrate_mode.isin(["force:fibroblasts_PDMS"] if mode == "ref" else ["n/a"])]
            return d.groupby("sample_dir").agg(mse=("mse", "mean"), r=("pixel_r", "mean"), rme=("rme", "mean"),
                                               ss=("ms_ssim", "mean"), mg=("mean_gt", "mean"), mp=("mean_pred", "mean"))
        spec = {"released": (["released_singlecell"], "ref"), "image_only_stage1": (["stage1_image_only"], None),
                "staged": (["s2f_staged_seed0", "s2f_staged_seed1"], "ref"),
                "staged_no_force": (["s2f_staged_no_force_seed0", "s2f_staged_no_force_seed1"], "ref"),
                "bf_only": (["s2f_no_substrate_seed0", "s2f_no_substrate_seed1"], None),
                "bf_only_s0": (["s2f_no_substrate_seed0"], None),
                "no_multiscale": (["s2f_no_substrate_no_multiscale_seed0"], None),
                "plain_bottleneck": (["s2f_no_substrate_bridge_plain_seed0"], None),
                "no_attention_gates": (["s2f_no_substrate_no_attention_gates_seed0"], None),
                "no_discriminator": (["s2f_no_substrate_no_gan_seed0", "s2f_no_substrate_no_gan_seed1"], None),
                "unet": (["unet_no_substrate_seed0", "unet_no_substrate_seed1"], None),
                "pix2pix": (["unet_gan_no_substrate_seed0", "unet_gan_no_substrate_seed1"], None),
                "mean_map": (["baseline_mean_map_singlecell"], None)}
        G = {k: M(*v) for k, v in spec.items()}
        comps = [("Released vs U-Net", "released", "unet"), ("Released vs pix2pix", "released", "pix2pix"),
                 ("Released vs image-only stage (i)", "released", "image_only_stage1"),
                 ("Retrained staged vs released (reproduction)", "staged", "released"),
                 ("Substrate channels: retrained staged vs BF only (same phase 1)", "staged", "bf_only"),
                 ("Stress consistency: staged vs staged without the term", "staged", "staged_no_force"),
                 ("Multi-scale input module", "bf_only_s0", "no_multiscale"),
                 ("Bottleneck context/attention modules", "bf_only_s0", "plain_bottleneck"),
                 ("Attention gates", "bf_only_s0", "no_attention_gates"),
                 ("PatchGAN discriminator", "bf_only", "no_discriminator"),
                 ("S2F (BF only) vs U-Net", "bf_only", "unet"), ("S2F (BF only) vs pix2pix", "bf_only", "pix2pix"),
                 ("S2F (BF only) vs mean map", "bf_only", "mean_map")]
        for name, a, b in comps:
            A, B = G[a], G[b]
            if not len(A) or not len(B):
                continue
            B = B.loc[A.index]
            r = dict(set=split, comparison=name, n=len(A))
            for k, c in [("r", "r"), ("RME", "rme"), ("MSE", "mse"), ("MS_SSIM", "ss")]:
                r[f"delta_{k}"] = (A[c] - B[c]).mean(); r[f"p_{k}"] = stats.wilcoxon(A[c], B[c]).pvalue
            r["delta_pred_GT"] = A.mp.sum() / A.mg.sum() - B.mp.sum() / B.mg.sum()
            r["p_pred_GT"] = stats.wilcoxon(A.mp / A.mg, B.mp / B.mg).pvalue
            out.append(r)
    return pd.DataFrame(out)


def sets_table():
    rows = []
    for man, role in {**SC_MANIFESTS, **SPH_MANIFESTS}.items():
        for r in read_manifest(manifest_path(man)):
            rows.append(dict(role=role, set=set_label(r), condition=r.get("condition", ""), cell_type=r.get("cell_type", ""),
                             coating=r.get("coating", ""), young_Pa=r.get("young_Pa", ""), manifest=man))
    d = pd.DataFrame(rows)
    return d.groupby(["role", "set", "condition", "cell_type", "coating", "young_Pa", "manifest"]).size() \
        .rename("images").reset_index()


SHEETS = ["README", "sets", "models", "tests", "test_images", "validation_images", "training_images", "all_models"]


def write_xlsx(out):
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        print("openpyxl not installed: results_csv/ written; run  pip install openpyxl  and then\n"
              "  python -m experiments.build_results_package --stage1 x --out <folder> --xlsx-only")
        return
    with pd.ExcelWriter(os.path.join(out, "results.xlsx"), engine="openpyxl") as w:
        for name in SHEETS:
            f = os.path.join(out, "results_csv", f"{name}.csv")
            if os.path.exists(f):
                pd.read_csv(f).to_excel(w, sheet_name=name, index=False)
    print("wrote", os.path.join(out, "results.xlsx"))


# ---------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage1", required=True, help="image-only checkpoint (last_checkpoint_final.pth)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(S2F_ROOT), "S2F_results_v2"))
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-dataset", action="store_true", help="do not copy the images into dataset/")
    ap.add_argument("--skip-png", action="store_true", help="metrics only (no panel/prediction images)")
    ap.add_argument("--with-tif", action="store_true", help="also write prediction_Pa.tif (32-bit map in Pa, ~4 MB each)")
    ap.add_argument("--with-staged", action="store_true", help="also copy the retrained staged generators")
    ap.add_argument("--limit", type=int, default=None, help="images per manifest (smoke test)")
    ap.add_argument("--xlsx-only", action="store_true", help="only (re)write results.xlsx from results_csv/")
    a = ap.parse_args()
    if a.xlsx_only:
        write_xlsx(os.path.abspath(a.out)); return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from PIL import Image
    from experiments.evaluate import CONFIG_PATH, released_models, stage1_model
    from experiments.figure3_panels import _has_pair, _segmentation
    from experiments.losses import PerImageWFMRMELoss
    from experiments.manifest_data import _find, make_loader
    from models.s2f_model import create_settings_channels
    from utils.metrics import TorchMetricsWrapper
    from utils.substrate_settings import compute_settings_normalization

    out = os.path.abspath(a.out); os.makedirs(out, exist_ok=True)
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rel = released_models(dev)
    models_sc = {"released": (rel["released_singlecell"]["model"], True, True),
                 "image_only": (stage1_model(a.stage1, dev)["model"], False, True)}
    models_sph = {"spheroid_GN": (rel["released_spheroid_GN"]["model"], False, True)}
    if "supervised_spheroid_FN" in rel:
        models_sph["spheroid_FN"] = (rel["supervised_spheroid_FN"]["model"], False, False)
    mw, norm, rme = TorchMetricsWrapper(device=dev), compute_settings_normalization(config_path=CONFIG_PATH), PerImageWFMRMELoss()
    est = _segmentation()
    tmpdir, rows, missing = tempfile.mkdtemp(), [], []

    # splits
    os.makedirs(os.path.join(out, "splits"), exist_ok=True)
    for man in list(SC_MANIFESTS) + list(SPH_MANIFESTS):
        shutil.copy2(manifest_path(man), os.path.join(out, "splits", f"{man}.csv"))
    sets = sets_table(); sets.to_csv(os.path.join(out, "splits", "sets.csv"), index=False)

    for kind, mans, models in [("single_cell", SC_MANIFESTS, models_sc), ("spheroid", SPH_MANIFESTS, models_sph)]:
        for man, role in mans.items():
            mrows = read_manifest(manifest_path(man))
            keep = [r for r in mrows if _has_pair(os.path.join(a.dataset_root, r["sample_dir"]))]
            missing += [r["sample_dir"] for r in mrows if r not in keep]
            if a.limit:
                keep = keep[:a.limit]
            print(f"{man}: {len(keep)}/{len(mrows)} samples", flush=True)
            if not keep:
                continue
            tmp = os.path.join(tmpdir, f"{man}.csv"); pd.DataFrame(keep).to_csv(tmp, index=False)
            loader = make_loader(tmp, a.dataset_root, batch_size=2, substrate_mode="reference" if kind == "single_cell" else "none")
            meta, k = loader.dataset.rows, 0
            with torch.no_grad():
                for b in loader:
                    x, y = b[0].to(dev), b[1].to(dev)
                    xs = torch.cat([x, create_settings_channels(b[4], norm, dev, x.shape, config_path=CONFIG_PATH)], 1) \
                        if kind == "single_cell" else x
                    preds = {}
                    for nm, (g, sub, tanh) in models.items():
                        p = g(xs if sub else x)
                        preds[nm] = (((p + 1) / 2) if tanh else p).clamp(0, 1)
                    for i in range(y.size(0)):
                        r = meta[k]; k += 1
                        sl, smp = set_label(r), sample_label(r)
                        yt = float(r.get("ythreshold_Pa") or (500 if kind == "single_cell" else 50))
                        t = y[i:i + 1]; tn = t[0, 0].cpu().numpy(); bf = x[i, 0].cpu().numpy()
                        if not a.skip_dataset:
                            dst = os.path.join(out, "dataset", kind, role, sl, smp); os.makedirs(dst, exist_ok=True)
                            src = os.path.join(a.dataset_root, r["sample_dir"])
                            for f in (_find(src, suffix="BF_001.tif"), _find(src, suffix="_gray.jpg")):
                                shutil.copy2(f, dst)
                        mg = est(tn).astype(bool); mg = mg if mg.any() else np.ones_like(tn, bool)
                        for nm, p in preds.items():
                            q = p[i:i + 1]; qn = q[0, 0].cpu().numpy()
                            d = dict(model=nm, kind=kind, role=role, set_label=sl, sample=smp, sample_dir=r["sample_dir"],
                                     manifest=man, condition=r.get("condition", ""), young_Pa=r.get("young_Pa", ""),
                                     bf_md5=r.get("bf_md5", ""), substrate_input="6 kPa reference" if models[nm][1] else "none",
                                     note=DUP_NOTE if r["sample_dir"] in DUP else "",
                                     mse=mw.compute_mse(t, q), pixel_r=float(np.corrcoef(tn.ravel(), qn.ravel())[0, 1]),
                                     rme=float(rme(q, t)), ms_ssim=mw.compute_ms_ssim(t, q),
                                     gt_field_mean_Pa=float(tn.mean()) * yt, pred_field_mean_Pa=float(qn.mean()) * yt,
                                     gt_region_mean_Pa=float(tn[mg].mean()) * yt, pred_region_mean_Pa=float(qn[mg].mean()) * yt,
                                     gt_p99_Pa=float(np.percentile(tn, 99)) * yt, pred_p99_Pa=float(np.percentile(qn, 99)) * yt,
                                     force_region_fraction=float(mg.mean()))
                            d["pred_gt_field"] = d["pred_field_mean_Pa"] / max(d["gt_field_mean_Pa"], 1e-9)
                            d["pred_gt_region"] = d["pred_region_mean_Pa"] / max(d["gt_region_mean_Pa"], 1e-9)
                            rows.append(d)
                            if not a.skip_png:
                                pdir = os.path.join(out, "predictions", nm, role, sl, smp)
                                warn = " | TRAINING IMAGE: not for evaluation" if role == "train" else ""
                                save_maps(pdir, bf, tn * yt, qn * yt, yt, d,
                                          f"{role} | {sl} | {smp} | {r.get('condition', '')} | {d['substrate_input']}{warn}",
                                          plt, Image, a.with_tif)
            print(f"  done {man}", flush=True)

    P = pd.DataFrame(rows)
    sheets = {
        "README": None,
        "sets": sets,
        "models": models_table(P),
        "tests": tests_table(os.path.join(RESULTS_DIR, "eval", "per_image.csv")),
        "test_images": P[P.role == "test"],
        "validation_images": P[P.role == "validation"],
        "training_images": P[P.role == "train"].assign(note=lambda d: "TRAINING IMAGE, not for evaluation. " + d.note),
    }
    ev = os.path.join(RESULTS_DIR, "eval", "per_image.csv")
    if os.path.exists(ev):
        E = pd.read_csv(ev)
        sheets["all_models"] = E.groupby(["model", "substrate_mode", "manifest", "set"], dropna=False).agg(
            n=("mse", "size"), MSE=("mse", "mean"), RME=("rme", "mean"), pixel_r=("pixel_r", "mean"),
            MS_SSIM=("ms_ssim", "mean"), mean_gt=("mean_gt", "sum"), mean_pred=("mean_pred", "sum")).reset_index() \
            .assign(pred_GT_field=lambda d: d.mean_pred / d.mean_gt).drop(columns=["mean_gt", "mean_pred"])
    try:
        commit = subprocess.run(["git", "-C", S2F_ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = "unknown"
    readme = pd.DataFrame({"item": [
        "Created", "Code version", "Primary model", "Substrate setting", "Second single-cell model", "Spheroid models",
        "Roles", "Metrics", "pred/GT", "Force region", "Pools", "Position_08", "Missing samples"],
        "description": [
        datetime.date.today().isoformat(), f"S2F repository commit {commit}",
        "Released S2F single-cell checkpoint (ckp_singlecell_GN.pth). All reported single-cell results use this model.",
        "6 kPa reference setting for every image (as in the paper and the web-app default).",
        "Image-only S2F = stage (i) of the released model (bright field only).",
        "spheroid_GN = paper model (Fig. 3G-I); spheroid_FN = smaller supervised model, web-app default.",
        "train = used to update the weights (predictions for completeness only, NOT for evaluation); validation = "
        "checkpoint selection only; test = independent experiments.",
        "Per image: MSE (normalised 0-1 maps), RME (force-weighted relative magnitude error, ref. [20]), pixel r "
        "(Pearson), MS-SSIM. Pooled values = mean over images.",
        "pred_GT_field = sum of predicted / sum of measured whole-field mean stress; pred_GT_region_geomean = geometric "
        "mean over images of predicted / measured mean stress in the force region detected on the GT map.",
        "Detected on the GT map with the web-app detector; the same region is used for GT and prediction.",
        "1 kPa pool = validation 1kPa_SET_I&II + test 1kPa_SET_III; 6 kPa pool = validation 6kPa_SET_VI + 6 kPa test "
        "sets (DMSO, Y-27632, TGF-beta, fibronectin); 10 kPa pool = validation 10kPa_SET_I (no 10 kPa test set); "
        "Test pool = all HFF test sets except U2OS.",
        DUP_NOTE + ". Counted once in Test and 6 kPa pools; left out of the DMSO and Y-27632 groups.",
        f"{len(missing)} manifest samples were not found under the dataset root" + (f" (first: {missing[0]})" if missing else "")]})
    sheets["README"] = readme
    os.makedirs(os.path.join(out, "results_csv"), exist_ok=True)
    for name, df in sheets.items():
        df.to_csv(os.path.join(out, "results_csv", f"{name}.csv"), index=False)
    write_xlsx(out)

    # checkpoints + model card
    ck = os.path.join(out, "checkpoints"); os.makedirs(ck, exist_ok=True)
    cks = {"released": os.path.join(S2F_ROOT, MODEL_INFO["released"][1]), "image_only": a.stage1,
           "spheroid_GN": os.path.join(S2F_ROOT, MODEL_INFO["spheroid_GN"][1]),
           "spheroid_FN": os.path.join(S2F_ROOT, MODEL_INFO["spheroid_FN"][1])}
    names = {"released": "ckp_singlecell_GN.pth", "image_only": "ckp_singlecell_image_only_stage1.pth",
             "spheroid_GN": "ckp_spheroid_GN.pth", "spheroid_FN": "ckp_spheroid_FN.pth"}
    if a.with_staged:
        for s in (0, 1):
            src = os.path.join(RUNS_DIR, f"s2f_staged_seed{s}", "best_checkpoint.pth")
            if os.path.exists(src):
                cks[f"retrained_staged_seed{s}"] = src; names[f"retrained_staged_seed{s}"] = f"s2f_staged_seed{s}.pth"
    M = sheets["models"]
    card = ["# Model card", "", f"Created {datetime.date.today().isoformat()}, S2F commit {commit}.", ""]
    for nm, src in cks.items():
        if not src or not os.path.exists(src):
            card.append(f"- **{nm}**: checkpoint not found ({src})"); continue
        dst = os.path.join(ck, names[nm])
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        info = MODEL_INFO.get(nm, (f"Retrained staged S2F, {nm[-5:]}", None, "bright field + 2 substrate channels"))
        line = f"- **{names[nm]}**: {info[0]}. Input: {info[2]}. SHA-256 {sha256(dst)}."
        sub = M[(M.model == nm) & M.set.isin(["Test pool", "Spheroid validation"])]
        if len(sub):
            s = sub.iloc[0]
            line += (f" {s.set} (n={int(s.n)}): r {s.pixel_r:.3f}, RME {s.RME:.3f}, MSE {s.MSE * 1e3:.2f}e-3, MS-SSIM {s.MS_SSIM:.3f}, "
                     f"pred/GT {s.pred_GT_field:.2f}.")
        card.append(line)
    card += ["", "Training: see the Methods (staged training) and experiments/configs.py; splits in splits/."]
    open(os.path.join(ck, "MODEL_CARD.md"), "w").write("\n".join(card) + "\n")

    # README + checksums
    open(os.path.join(out, "README.md"), "w").write("\n".join([
        "# S2F results package (v2)", "",
        f"Created {datetime.date.today().isoformat()} with experiments/build_results_package.py (S2F commit {commit}).", "",
        "- `dataset/`: image pairs (bright field, GT traction-stress map) by role and set; `splits/`: the split files.",
        "- `predictions/<model>/<role>/<set>/<sample>/`: panel.png (BF | GT | prediction with metrics), prediction.png,"
        " (prediction_Pa.tif, 32-bit map in Pa, only if built with --with-tif).",
        "- `results.xlsx` / `results_csv/`: README (definitions), sets, models, tests, per-image sheets, all_models.",
        "- `checkpoints/`: model files and MODEL_CARD.md (SHA-256, inputs, headline metrics).", "",
        "Only validation and test images are used for any reported number; training-image predictions are included "
        "for completeness and are marked as such.", ""]))
    with open(os.path.join(out, "CHECKSUMS.sha256"), "w") as f:
        for root, _, files in os.walk(out):
            for fn in sorted(files):
                if fn == "CHECKSUMS.sha256":
                    continue
                p = os.path.join(root, fn)
                f.write(f"{sha256(p)}  {os.path.relpath(p, out)}\n")
    print(f"done: {out} ({len(P)} image x model rows; {len(missing)} samples missing)")


if __name__ == "__main__":
    main()
