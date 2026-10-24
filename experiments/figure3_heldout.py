"""Recommended Fig. 3: held-out images only, agreement shown with concordance statistics.

Cohorts (ML-dataset labels, no training images; Position_08 counted in neither drug group):
  1 kPa   test SET III (17)  + validation SET I&II (15, open markers)
  10 kPa  validation SET I (20, open markers; there is no 10 kPa test set)
  DMSO    test Control SET I (27) + test SET II (10)
  Y-27632 test SET I (16) + test SET II (19)
  TGF-beta test SET I (15)

Panels per model
  B / E  per-image MSE, pixel correlation r and RME (mean +/- SD, one dot per image)
  C / F  mean traction stress in the force region, GT vs P per cell (paired lines), with
         P/GT (geometric mean), Lin's CCC, Bland-Altman bias [95% limits] and a TOST equivalence
         test on log(P/GT) with margin 0.80-1.25
  D / G  effect concordance: fold change of each condition vs the reference (1 kPa; DMSO)
         in GT and in P, with bootstrap 95% CI (cells resampled within group)
  H      spheroids (released GN model, 28 validation spheroids): MSE, r, RME

Usage (recommended: per-image values from experiments.predict_panels, both models in one pass)
  python -m experiments.predict_panels --stage1 <last_checkpoint_final.pth>
  python -m experiments.figure3_heldout plot
  -> results/figure3_heldout/fig3_heldout_<model>.png/.pdf, fig3_heldout_stats.csv,
     fig3_heldout_cells.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

S2F_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if S2F_ROOT not in sys.path:
    sys.path.insert(0, S2F_ROOT)

from experiments import figure3_panels as F  # noqa: E402
from experiments.layout import DEFAULT_DATASET_ROOT, RESULTS_DIR  # noqa: E402

OUT = os.path.join(RESULTS_DIR, "figure3_heldout")
HELDOUT = ["alltune_val", "test_fibro_1kPa_setIII", "test_fibro_DMSO_setI", "test_fibro_DMSO_setII",
           "test_fibro_Y27632_setI", "test_fibro_Y27632_setII", "test_fibro_TGFb_setI"]
MODELS = {"released": "Released S2F (BF + substrate channels, 6 kPa reference setting)", "image_only": "Image-only S2F (BF)"}
STIFF, DRUGS = ["1 kPa", "10 kPa"], ["DMSO", "Y-27632", "TGF-β"]
MARGIN = (0.80, 1.25)
COL = F.COL


def cohort(pi):
    c = F.cohorts_mine(pi, keep_dup=False)
    c["role"] = np.where(c.manifest == "alltune_val", "validation", "test")
    return c


# ------------------------------------------------------------------------------------------ stats
def ccc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return 2 * np.cov(x, y, bias=True)[0, 1] / (x.var() + y.var() + (x.mean() - y.mean()) ** 2)


def agreement(d):
    from scipy import stats
    g, p = d.region_mean_gt_Pa.values, d.region_mean_pred_Pa.values
    lr = np.log(p / g)
    n = len(lr)
    se = lr.std(ddof=1) / np.sqrt(n)
    lo, hi = np.log(MARGIN[0]), np.log(MARGIN[1])
    p_lo = 1 - stats.t.cdf((lr.mean() - lo) / se, n - 1)
    p_hi = stats.t.cdf((lr.mean() - hi) / se, n - 1)
    diff = p - g
    ci = np.exp(lr.mean() + np.array([-1, 1]) * stats.t.ppf(0.975, n - 1) * se)
    return dict(n=n, gt_mean_Pa=g.mean(), pred_mean_Pa=p.mean(), ratio_gmean=float(np.exp(lr.mean())),
                ratio_ci_low=ci[0], ratio_ci_high=ci[1], ccc=ccc(g, p), pearson=float(np.corrcoef(g, p)[0, 1]),
                ba_bias_Pa=diff.mean(), ba_loa_low=diff.mean() - 1.96 * diff.std(ddof=1),
                ba_loa_high=diff.mean() + 1.96 * diff.std(ddof=1), p_tost=float(max(p_lo, p_hi)),
                p_wilcoxon=float(stats.wilcoxon(g, p).pvalue))


def fold(c, grp, ref, col, nboot=2000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = c[c.group == ref][col].values, c[c.group == grp][col].values
    est = b.mean() / a.mean()
    bs = [rng.choice(b, len(b)).mean() / rng.choice(a, len(a)).mean() for _ in range(nboot)]
    return est, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def all_stats(c, model):
    from scipy import stats
    rows = []
    for g in STIFF + DRUGS:
        d = c[c.group == g]
        if len(d) > 2:
            rows.append(dict(model=model, kind="agreement GT vs P (cells)", group=g, **agreement(d)))
    for groups, ref in [(STIFF, "1 kPa"), (DRUGS, "DMSO")]:
        for g in groups:
            if g == ref or not ((c.group == g).any() and (c.group == ref).any()):
                continue
            r = dict(model=model, kind=f"effect vs {ref}", group=g, n=int((c.group == g).sum()))
            for col, lab in [("region_mean_gt_Pa", "gt"), ("region_mean_pred_Pa", "pred")]:
                e, lo, hi = fold(c, g, ref, col)
                r.update({f"fold_{lab}": e, f"fold_{lab}_ci_low": lo, f"fold_{lab}_ci_high": hi,
                          f"p_mannwhitney_{lab}": stats.mannwhitneyu(c[c.group == ref][col], c[c.group == g][col]).pvalue})
            rows.append(r)
    return rows


# ------------------------------------------------------------------------------------------ plots
def _metrics(axs, c, groups, show_role=True):
    rng = np.random.default_rng(0)
    for ax, (col, lab) in zip(axs, [("mse", "MSE"), ("pixel_r", "Pixel correlation r"), ("rme", "RME")]):
        for i, g in enumerate(groups):
            d = c[c.group == g].dropna(subset=[col])
            if not len(d):
                ax.text(i, 0, "n.d.", ha="center", va="bottom", fontsize=6); continue
            ax.bar(i, d[col].mean(), yerr=d[col].std(), width=0.65, color=COL[g], alpha=0.3, edgecolor=COL[g], lw=0.8,
                   error_kw=dict(lw=0.7, capsize=2))
            val = (d.role == "validation").values if show_role and "role" in d else np.zeros(len(d), bool)
            j = rng.uniform(-0.2, 0.2, len(d))
            ax.scatter(i + j[~val], d[col].values[~val], s=3, color=COL[g], lw=0, zorder=3)
            ax.scatter(i + j[val], d[col].values[val], s=5, facecolor="white", edgecolor=COL[g], lw=0.5, zorder=3)
        ax.set_xticks(range(len(groups)), groups, rotation=35 if len(groups) > 2 else 0,
                      ha="right" if len(groups) > 2 else "center", rotation_mode="anchor")
        ax.set_title(lab, fontsize=7, loc="left")
        ax.margins(x=0.1)
        if col == "pixel_r":
            ax.set_ylim(0, 1)


def _pairs(ax, c, groups, st):
    rng = np.random.default_rng(1)
    for i, g in enumerate(groups):
        d = c[c.group == g]
        if not len(d):
            ax.text(i * 2.4 + 0.5, 0.02, f"{g}\nn.d.", transform=ax.get_xaxis_transform(), ha="center", fontsize=6)
            continue
        x0, x1 = i * 2.4, i * 2.4 + 1
        j = rng.uniform(-0.12, 0.12, len(d))
        for (_, r), jj in zip(d.iterrows(), j):
            ax.plot([x0 + jj, x1 + jj], [r.region_mean_gt_Pa, r.region_mean_pred_Pa], color=COL[g], lw=0.3, alpha=0.35)
        for x, col in [(x0, "region_mean_gt_Pa"), (x1, "region_mean_pred_Pa")]:
            val = (d.role == "validation").values
            ax.scatter(x + j[~val], d[col].values[~val], s=4, color=COL[g], lw=0, zorder=3)
            ax.scatter(x + j[val], d[col].values[val], s=6, facecolor="white", edgecolor=COL[g], lw=0.5, zorder=3)
            ax.plot([x - 0.3, x + 0.3], [d[col].mean()] * 2, color="k", lw=1.2, zorder=4)
        a = [s for s in st if s["group"] == g and s["kind"].startswith("agreement")]
        if a:
            a = a[0]
            eq = "equivalent" if a["p_tost"] < 0.05 else "not shown equivalent"
            ax.text(i * 2.4 + 0.5, 1.0, f"P/GT {a['ratio_gmean']:.2f}\nCCC {a['ccc']:.2f}\nbias {a['ba_bias_Pa']:+.0f} Pa\n{eq}",
                    transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=5.5)
        ax.text(i * 2.4 + 0.5, -0.13, f"{g} (n={len(d)})", transform=ax.get_xaxis_transform(), ha="center",
                va="top", fontsize=6)
    ax.set_xticks([k for i in range(len(groups)) for k in (i * 2.4, i * 2.4 + 1)], ["GT", "P"] * len(groups))
    ax.set_ylabel("Mean traction stress,\nforce region (Pa)")
    ax.set_ylim(bottom=0)


def _effects(ax, st, groups, ref):
    rows = [s for s in st if s["kind"] == f"effect vs {ref}" and s["group"] in groups]
    for k, r in enumerate(rows):
        for dy, lab, colr in [(-0.12, "gt", "0.25"), (0.12, "pred", COL[r["group"]])]:
            e, lo, hi = r[f"fold_{lab}"], r[f"fold_{lab}_ci_low"], r[f"fold_{lab}_ci_high"]
            ax.errorbar(e, k + dy, xerr=[[e - lo], [hi - e]], fmt="o" if lab == "gt" else "s", ms=3.5, color=colr,
                        mfc="white" if lab == "gt" else colr, lw=0.8, capsize=1.5)
    ax.axvline(1, color="0.6", lw=0.6, ls="--")
    ax.set_xscale("log")
    ax.set_yticks(range(len(rows)), [f"{r['group']} / {ref}" for r in rows])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("Fold change in mean stress")
    ax.invert_yaxis()
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec="0.25", ms=4, label="GT"),
                       Line2D([], [], marker="s", ls="", mfc="0.5", mec="0.5", ms=4, label="P")],
              frameon=False, fontsize=6, loc="lower right")


def load_predictions(path):
    """predictions_metrics.csv (experiments.predict_panels) -> {model: per-image frame with the columns used here}."""
    d = pd.read_csv(path).rename(columns={"gt_region_mean_Pa": "region_mean_gt_Pa", "pred_region_mean_Pa": "region_mean_pred_Pa"})
    return {m: g.drop(columns="model") for m, g in d.groupby("model")}


def plot(models, predictions=None):
    import matplotlib.pyplot as plt
    F._style()
    ev = pd.read_csv(F.EVAL)
    sph = ev[(ev.model == "released_spheroid_GN") & (ev.manifest == "spheroid_val")].assign(group="spheroid", role="validation")
    allst, cells = [], []
    preds = load_predictions(predictions) if predictions else {}
    os.makedirs(OUT, exist_ok=True)
    for m in models:
        if m in preds:
            pi = preds[m]
        else:
            f = os.path.join(OUT, f"per_image_fig3_{m}.csv")
            if not os.path.exists(f):
                print("missing", f); continue
            pi = pd.read_csv(f)
        c = cohort(pi)
        cells.append(c.assign(model=m))
        st = all_stats(c, m)
        allst += st
        fig = plt.figure(figsize=(10, 8.2))
        gs = fig.add_gridspec(3, 3, width_ratios=[1.35, 1.25, 0.8], hspace=0.95, wspace=0.42,
                              left=0.06, right=0.985, top=0.9, bottom=0.07)
        for r, (lb, groups, ref) in enumerate([("BCD", STIFF, "1 kPa"), ("EFG", DRUGS, "DMSO")]):
            sub = gs[r, 0].subgridspec(1, 3, wspace=0.5)
            axs = [fig.add_subplot(sub[0, k]) for k in range(3)]
            _metrics(axs, c, groups)
            axs[0].text(-0.6, 1.25, lb[0], transform=axs[0].transAxes, fontsize=10, fontweight="bold")
            ax = fig.add_subplot(gs[r, 1]); _pairs(ax, c, groups, st)
            ax.text(-0.25, 1.25, lb[1], transform=ax.transAxes, fontsize=10, fontweight="bold")
            ax = fig.add_subplot(gs[r, 2]); _effects(ax, st, groups, ref)
            ax.set_title("Condition effect, GT vs P", fontsize=7, loc="left")
            ax.text(-0.45, 1.25, lb[2], transform=ax.transAxes, fontsize=10, fontweight="bold")
        sub = gs[2, 0].subgridspec(1, 3, wspace=0.5)
        axs = [fig.add_subplot(sub[0, k]) for k in range(3)]
        _metrics(axs, sph, ["spheroid"], show_role=False)
        for a in axs:
            a.set_xticks([0], [f"LS174T (n={len(sph)})"])
        axs[0].text(-0.6, 1.25, "H", transform=axs[0].transAxes, fontsize=10, fontweight="bold")
        n_tgf = int((c.group == "TGF-β").sum())
        note = ("Held-out images only (no training images). Filled dots: test images; open dots: validation images "
                "(1 kPa SET I&II; 10 kPa SET I, no 10 kPa test set exists). Position_08 excluded (same image in DMSO SET II "
                "and Y-27632 SET II). Stress = mean traction stress in the force region detected on the GT map "
                "(web-app detector), same region for GT and P. P/GT = geometric mean ratio; CCC = Lin's concordance; "
                "bias = Bland-Altman mean difference; equivalence = TOST on log(P/GT), margin 0.80-1.25. Effects: ratio of "
                "group means, bootstrap 95% CI. H: released spheroid GN model, 28 validation spheroids.")
        if n_tgf == 0:
            note += " TGF-β images were not available for this run."
        import textwrap
        fig.text(0.01, 0.005, "\n".join(textwrap.wrap(note, 230)), fontsize=5.5, color="0.3", va="bottom")
        fig.suptitle(f"Fig. 3 (held-out images): {MODELS[m]}", x=0.01, ha="left", fontsize=9)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(OUT, f"fig3_heldout_{m}.{ext}"), dpi=300)
        plt.close(fig)
    pd.DataFrame(allst).to_csv(os.path.join(OUT, "fig3_heldout_stats.csv"), index=False)
    if cells:
        pd.concat(cells).to_csv(os.path.join(OUT, "fig3_heldout_cells.csv"), index=False)
    print("wrote", OUT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["infer", "plot"])
    ap.add_argument("--model", choices=list(MODELS), default="released")
    ap.add_argument("--stage1", default=None)
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--device", default=None)
    ap.add_argument("--models", default="released,image_only", help="models to plot")
    ap.add_argument("--predictions", default=os.path.join(RESULTS_DIR, "predictions", "predictions_metrics.csv"),
                    help="per-image CSV of experiments.predict_panels (used when present)")
    a = ap.parse_args()
    if a.step == "infer":
        F.infer(a.dataset_root, a.device, model=a.model, stage1=a.stage1, substrate="nominal",
                manifests=HELDOUT, tag=a.model, out=OUT)
    else:
        plot([m for m in a.models.split(",") if m], a.predictions if os.path.exists(a.predictions) else None)


if __name__ == "__main__":
    main()
