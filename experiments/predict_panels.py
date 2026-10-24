"""Per-image prediction panels for the candidate single-cell models.

For every image of the chosen manifests, one PNG with: bright-field | GT | released S2F (BF + substrate) | image-only S2F (BF). Heatmaps share one colour scale in Pa (0 to ythreshold, 500 Pa for all HFF
sets), with MSE, pixel r, RME, MS-SSIM, mean stress in the force region and P/GT printed under each prediction.
Force region = region detected on the GT map (web-app detector), used for both GT and P, so the GT value does not
depend on the model. pred_own_region_mean_Pa = P in the region detected on P itself (what the web app reports).
Also writes predictions_metrics.csv (one row per image x model).

  python -m experiments.predict_panels --stage1 <path>/last_checkpoint_final.pth
      [--manifests test_fibro_1kPa_setIII,...] [--dataset-root ...] [--out results/predictions]
Default manifests: all independent test sets plus alltune_val.
"""
import argparse
import os
import sys
import tempfile

import numpy as np
import pandas as pd

S2F_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if S2F_ROOT not in sys.path:
    sys.path.insert(0, S2F_ROOT)

from experiments.layout import DEFAULT_DATASET_ROOT, RESULTS_DIR, manifest_path, read_manifest  # noqa: E402

DEFAULT = ["test_fibro_1kPa_setIII", "test_fibro_DMSO_setI", "test_fibro_DMSO_setII", "test_fibro_Y27632_setI",
           "test_fibro_Y27632_setII", "test_fibro_TGFb_setI", "test_fibro_fibronectin_setI", "test_u2os_6kPa",
           "alltune_val"]
NAMES = {"released": "Released S2F (BF + substrate)", "image_only": "Image-only S2F (BF)"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage1", required=True, help="image-only checkpoint (last_checkpoint_final.pth)")
    ap.add_argument("--manifests", default=",".join(DEFAULT))
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "predictions"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-png", action="store_true", help="metrics CSV only")
    ap.add_argument("--substrate", choices=["reference", "nominal"], default="reference",
                    help="substrate setting for the released model: 'reference' = 6 kPa reference setting for every "
                         "image (web-app default, used for all results in the paper); 'nominal' = true setting of each set")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from experiments.evaluate import CONFIG_PATH, released_models, stage1_model
    from experiments.figure3_panels import _has_pair, _segmentation
    from experiments.losses import PerImageWFMRMELoss
    from experiments.manifest_data import make_loader
    from models.s2f_model import create_settings_channels
    from utils.metrics import TorchMetricsWrapper
    from utils.substrate_settings import compute_settings_normalization

    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    models = {"released": (released_models(dev)["released_singlecell"]["model"], True),
              "image_only": (stage1_model(a.stage1, dev)["model"], False)}
    mw, norm, rme = TorchMetricsWrapper(device=dev), compute_settings_normalization(config_path=CONFIG_PATH), PerImageWFMRMELoss()
    est = _segmentation()
    rows, tmpdir = [], tempfile.mkdtemp()
    os.makedirs(a.out, exist_ok=True)
    for man in [m for m in a.manifests.split(",") if m]:
        mrows = read_manifest(manifest_path(man))
        keep = [r for r in mrows if _has_pair(os.path.join(a.dataset_root, r["sample_dir"]))]
        print(f"{man}: {len(keep)}/{len(mrows)} samples found", flush=True)
        if not keep:
            continue
        tmp = os.path.join(tmpdir, f"{man}.csv"); pd.DataFrame(keep).to_csv(tmp, index=False)
        loader = make_loader(tmp, a.dataset_root, batch_size=2, substrate_mode=a.substrate)
        meta_rows, k = loader.dataset.rows, 0
        pdir = os.path.join(a.out, man); os.makedirs(pdir, exist_ok=True)
        with torch.no_grad():
            for b in loader:
                x, y = b[0].to(dev), b[1].to(dev)
                xs = torch.cat([x, create_settings_channels(b[4], norm, dev, x.shape, config_path=CONFIG_PATH)], 1)
                preds = {nm: ((g(xs if sub else x) + 1) / 2).clamp(0, 1) for nm, (g, sub) in models.items()}
                for i in range(y.size(0)):
                    r = meta_rows[k]; k += 1
                    yt = float(r.get("ythreshold_Pa") or 500)
                    t = y[i:i + 1]; tn = t[0, 0].cpu().numpy()
                    mg = est(tn).astype(bool)           # force region detected on the GT map (model-independent)
                    mg = mg if mg.any() else np.ones_like(tn, bool)
                    res = {}
                    for nm, p in preds.items():
                        q = p[i:i + 1]; qn = q[0, 0].cpu().numpy()
                        mp = est(qn).astype(bool)           # force region detected on the prediction (as in the web app)
                        mp = mp if mp.any() else np.ones_like(qn, bool)
                        d = dict(manifest=man, sample_dir=r["sample_dir"], set=r.get("set", ""),
                                 condition=r.get("condition", ""), young_Pa=r.get("young_Pa", ""),
                                 substrate_input=(("fibroblasts_PDMS" if a.substrate == "reference" else r.get("substrate_nominal", ""))
                                                  if models[nm][1] else "none"),
                                 model=nm, mse=mw.compute_mse(t, q), pixel_r=float(np.corrcoef(tn.ravel(), qn.ravel())[0, 1]),
                                 rme=float(rme(q, t)), ms_ssim=mw.compute_ms_ssim(t, q),
                                 gt_region_mean_Pa=float(tn[mg].mean()) * yt, pred_region_mean_Pa=float(qn[mg].mean()) * yt,
                                 pred_own_region_mean_Pa=float(qn[mp].mean()) * yt, gt_region_frac=float(mg.mean()),
                                 gt_field_mean_Pa=float(tn.mean()) * yt, pred_field_mean_Pa=float(qn.mean()) * yt,
                                 gt_p99_Pa=float(np.percentile(tn, 99)) * yt, pred_p99_Pa=float(np.percentile(qn, 99)) * yt)
                        d["pred_gt_region"] = d["pred_region_mean_Pa"] / max(d["gt_region_mean_Pa"], 1e-9)
                        rows.append(d); res[nm] = (qn, d)
                    if a.no_png:
                        continue
                    bf = x[i, 0].cpu().numpy()
                    fig, axs = plt.subplots(1, 4, figsize=(12, 3.6), gridspec_kw=dict(wspace=0.04))
                    axs[0].imshow(bf, cmap="gray"); axs[0].set_title("Bright-field", fontsize=8)
                    im = axs[1].imshow(tn * yt, cmap="jet", vmin=0, vmax=yt)
                    axs[1].set_title(f"GT (TFM)\nmean in force region {res['released'][1]['gt_region_mean_Pa']:.0f} Pa", fontsize=8)
                    for ax, nm in zip(axs[2:], ["released", "image_only"]):
                        qn, d = res[nm]
                        ax.imshow(qn * yt, cmap="jet", vmin=0, vmax=yt)
                        ax.set_title(f"{NAMES[nm]}\nmean {d['pred_region_mean_Pa']:.0f} Pa (P/GT {d['pred_gt_region']:.2f})", fontsize=8)
                        ax.text(0.5, -0.04, f"MSE {d['mse']:.4f} | r {d['pixel_r']:.2f} | RME {d['rme']:.2f} | MS-SSIM {d['ms_ssim']:.2f}",
                                transform=ax.transAxes, ha="center", va="top", fontsize=7)
                    for ax in axs:
                        ax.set_xticks([]); ax.set_yticks([])
                    cb = fig.colorbar(im, ax=axs[1:], fraction=0.015, pad=0.01); cb.set_label("Traction stress (Pa)", fontsize=7)
                    cb.ax.tick_params(labelsize=6)
                    sub = ("released model with the 6 kPa reference setting" if a.substrate == "reference"
                           else f"released model with the true setting ({r.get('substrate_nominal', '')})")
                    fig.suptitle(f"{man} | {r['sample_dir']} | {r.get('condition', '')} | {sub}", fontsize=8, x=0.01, ha="left")
                    fig.savefig(os.path.join(pdir, os.path.basename(r["sample_dir"]) + ".png"), dpi=110, bbox_inches="tight")
                    plt.close(fig)
        print(f"  done {man}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "predictions_metrics.csv"), index=False)
    s = df.groupby(["manifest", "model"])[["mse", "pixel_r", "rme", "ms_ssim", "pred_gt_region"]].mean().round(4)
    s.to_csv(os.path.join(a.out, "predictions_summary.csv"))
    print(s.to_string())


if __name__ == "__main__":
    main()
