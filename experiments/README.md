# S2F experiments: data audit, evaluation, baselines and ablations

This folder holds the code for every number reported in the paper beyond the training of the released checkpoints:
- the dataset audit and split manifests;
- the evaluation of the released models;
- the baselines and ablation study.

Run every command from the repository root (`Shape2Force/`); a GPU is used automatically when available.

```
experiments/
  layout.py           dataset layout, per-set metadata (Young's modulus, TFM settings), manifest helpers
  prepare_dataset.py  audit legacy data vs dataset/, copy missing images, write splits/*.csv
  manifest_data.py    data loader driven by a split manifest (same preprocessing as data/cell_dataset.py)
  ablation_models.py  plain U-Net and S2F generator variants
  losses.py           per-image WFM-RME and weighted MSE + soft Dice losses (both used for the released model)
  configs.py          shared training protocol + one entry per variant
  train.py            train one variant
  evaluate.py         evaluate released checkpoints / baselines / trained runs on all manifests
  run_all.py          run the steps above in order
  splits/             split manifests (tracked in git)
  runs/               training outputs (git-ignored)
  results/            audit and evaluation tables
```

## Data

`dataset/` (git-ignored, Zenodo record 10.5281/zenodo.18457879) contains:
- `singlecell/train` (236), `singlecell/test` (46);
- `spheroid/train` (129), `spheroid/test` (28).

The released single-cell model was trained on a larger split (`all_tune`: 280 training / 81 validation pairs, 1, 6 and 10 kPa). The paper also reports independent test sets. `prepare_dataset.py` reconstructs all of this from the original data tree and adds the missing images to `dataset/`:

```
dataset/singlecell_extra/<set>/<sample>/      all_tune images that are not in singlecell/ (1 and 10 kPa)
dataset/singlecell_external/<set>/<sample>/   independent test sets (DMSO, Y-27632, TGF-beta, fibronectin, ...)
```

Each manifest in `splits/` lists one sample per row:
- the sample folder relative to `dataset/`;
- condition, Young's modulus and TFM settings: colour limit of the heatmap (`ythreshold_Pa`, 500 Pa for single cells and 200 Pa for spheroids) and pixel-size setting;
- the substrate setting the original code fed to the network (`substrate_legacy`) and the nominal one (`substrate_nominal`);
- the MD5 of the bright-field image.

## Steps

```bash
# 1. audit only: what is in dataset/ vs the original tree, all_tune composition, leakage check
python -m experiments.run_all --legacy-root <path>/cellforce/dataset --steps audit

# 2. complete dataset/ and write splits/*.csv (then commit experiments/splits/)
python -m experiments.run_all --legacy-root <path>/cellforce/dataset --steps prepare

# 3. released checkpoints, substrate sweep, input-free baseline, image-only stage-1 model
python -m experiments.run_all --steps eval_released --stage1 <path>/outputs/checkpoints/last/last_checkpoint_final.pth

# 4. baselines / ablations (S2F about 7 h, U-Net about 2.5 h each on an RTX A6000; resumable), then their evaluation
python -m experiments.run_all --steps train,eval_runs --variants s2f_full,unet,unet_gan,s2f_no_gan,s2f_no_substrate,s2f_no_force
```

A single run: `python -m experiments.train --variant unet --seed 0`.

### GPUs and run time

- `--steps train` runs one job per GPU in parallel (default `--gpus 0,1`). Each job sees only its own GPU (`CUDA_VISIBLE_DEVICES`), and the next variant starts as soon as a GPU is free.
- A model is never split across GPUs: with batch size 2, that would harm batch-norm/GAN training.
- `train.py` refuses to run without CUDA unless `--allow-cpu` is given.
- Augmentation runs in `--num-workers` DataLoader processes (default 2 per job; use 0 if RAM is short).
- Finished runs (and finished phases) are skipped and interrupted runs resume from the phase's `last_checkpoint.pth`, so the command can simply be re-run.
- Run time on an RTX A6000: S2F about 3 min per epoch, i.e. about 7 h per variant (100 + 40 epochs); U-Net about 2.5 h. On two GPUs the 6 core variants take about 15 h (`s2f_no_force` reuses phase 1 of `s2f_full`).
- Progress is logged to `experiments/runs/<variant>_seed<seed>/train.log` (validation pixel r and WFM-RME every epoch) and `phase*/history.json`.

For a quick check that everything runs, add `--epochs 1 --max-batches 2` (1 epoch per phase; add `--allow-cpu` without a GPU).

## Training protocol (all variants)

Every variant is trained from scratch on `alltune_train` (280 pairs), batch size 2, 1024 × 1024 px, the augmentations of `data/augmentations.py`, Adam (β = 0.5/0.999) with cosine decay after 10 % linear warm-up, in two phases that mirror the staged training of the released model:

| phase | epochs | reconstruction loss | other terms | lr | starts from |
|---|---|---|---|---|---|
| 1 localisation | 100 | 100 × weighted MSE + soft Dice (pos_weight 3, threshold 0.2, Dice 0.5) | 1 × adversarial | 1e-4 | random weights |
| 2 magnitude | 40 | 10 × per-image WFM-RME | 1 × adversarial + 100 × MSE of mean intensity | 1e-5 | best phase-1 generator |

Each phase uses a new discriminator and optimiser and keeps the checkpoint with the lowest validation loss of that phase on `alltune_val` (81 pairs); the final model is the best phase-2 checkpoint. WFM-RME alone does not learn to localise the forces from random weights (a first protocol that used it from the start gave pixel r ≈ 0), which is also why the released model was first trained with the weighted MSE + Dice loss. Variants change one component each (`configs.py`):

| variant | change |
|---|---|
| `s2f_full` | none (from-scratch counterpart of the released model) |
| `unet` | plain U-Net (Ronneberger et al. 2015), no attention modules, no discriminator |
| `unet_gan` | plain U-Net + PatchGAN (pix2pix-style) |
| `mask_only` | binary cell mask instead of bright field (needs `cell_mask.tif` in each sample folder) |
| `s2f_no_gan` | no discriminator (λ_GAN = 0) |
| `s2f_no_force` | no stress-consistency term (λ_force = 0 in phase 2; phase 1 is identical to `s2f_full` and reused) |
| `s2f_gan_0.1`, `s2f_gan_5` | λ_GAN = 0.1 / 5 |
| `s2f_no_substrate` | bright field only (no substrate channels) |
| `s2f_nominal_substrate` | substrate channels from the nominal Young's modulus of each image |
| `s2f_no_multiscale` | no multi-scale input module |
| `s2f_bridge_no_cbam` | bottleneck without CBAM |
| `s2f_bridge_plain` | bottleneck without CBAM, Global Context Module and hierarchical attention |
| `s2f_no_attention_gates` | skip connections without attention gates |

The released checkpoint was obtained by staged fine-tuning from earlier checkpoints, with substrate channels added only in its last two stages (see Methods). `evaluate.py` reports it alongside the from-scratch variants.

## Metrics

`evaluate.py` computes every metric per image and reports the mean and SD across images:
- MSE, MS-SSIM and pixel-wise Pearson r;
- per-image WFM-RME;
- mean traction stress in Pa (heatmap value × colour limit);
- the fraction of saturated ground-truth pixels;
- inference time.

Its outputs are `results/eval/per_image.csv`, `summary.csv` and `model_params.csv`.
