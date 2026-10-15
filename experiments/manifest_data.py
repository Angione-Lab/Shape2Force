"""
Manifest-based data loading. Uses exactly the preprocessing of the S2F package
(``data.cell_dataset.load_image`` -> grayscale, resize to 1024, /255; ``ImageDataset``;
``AdvancedAugmentations``), so results are comparable with the released models.

substrate_mode (single-cell only) decides which substrate setting builds the two substrate channels:
  'legacy'    what the original training code fed (derived from the all_tune folder name) - default,
              reproduces the conditions under which the released model was trained
  'nominal'   the true Young's modulus / TFM pixel-size setting of each sample
  'reference' the 6 kPa reference setting (fibroblasts_PDMS) for every sample
  'none'      no metadata (models without substrate channels)
input_mode:
  'bf'        bright-field image (default)
  'mask'      binary cell mask (file cell_mask.tif/png in the sample folder) - mask-only baseline
"""
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.cell_dataset import ImageDataset, load_image
from experiments.layout import DEFAULT_DATASET_ROOT, MASK_NAMES, manifest_path, read_manifest


def _find(folder, suffix=None, names=None):
    for f in sorted(os.listdir(folder)):
        if (suffix and f.endswith(suffix)) or (names and f in names):
            return os.path.join(folder, f)
    return None


def load_manifest_pairs(manifest, dataset_root=DEFAULT_DATASET_ROOT, img_size=1024,
                        substrate_mode="legacy", input_mode="bf"):
    rows = read_manifest(manifest if os.path.isfile(manifest) else manifest_path(manifest))
    pairs = []
    for r in rows:
        d = os.path.join(dataset_root, r["sample_dir"])
        bf = _find(d, suffix="BF_001.tif")
        hm = _find(d, suffix="_gray.jpg")
        if bf is None or hm is None:
            raise FileNotFoundError(f"missing BF/heatmap in {d}")
        if input_mode == "mask":
            mp = _find(d, names=MASK_NAMES)
            if mp is None:
                raise FileNotFoundError(f"input_mode='mask' but no {MASK_NAMES} in {d}")
            x = (load_image(mp, (img_size, img_size)) > 0.5).astype(np.float32)
        else:
            x = load_image(bf, (img_size, img_size))
        y = load_image(hm, (img_size, img_size))
        if substrate_mode == "none":
            pairs.append((x, y, (0, 0)))
            continue
        sub = {"legacy": r.get("substrate_legacy"), "nominal": r.get("substrate_nominal"),
               "reference": "fibroblasts_PDMS"}[substrate_mode] or "fibroblasts_PDMS"
        pairs.append((x, y, (0, 0), {"folder_name": os.path.basename(d), "substrate": sub,
                                     "root_folder": os.path.dirname(r["sample_dir"]), "set": r.get("set", "")}))
    return pairs, rows


def _worker_init(worker_id):
    """Distinct, reproducible numpy seed per worker (augmentations use np.random). Module-level for Windows spawn."""
    np.random.seed(torch.initial_seed() % 2 ** 32)


class _MaskDataset(ImageDataset):
    """Re-binarise the input after augmentation (intensity augmentations are meaningless for masks)."""
    def __getitem__(self, idx):
        out = list(super().__getitem__(idx))
        out[0] = (out[0] > 0.5).float()
        return tuple(out)


def make_loader(manifest, dataset_root=DEFAULT_DATASET_ROOT, batch_size=2, img_size=1024, augment=False,
                shuffle=False, substrate_mode="legacy", input_mode="bf", seed=None, num_workers=0):
    pairs, rows = load_manifest_pairs(manifest, dataset_root, img_size, substrate_mode, input_mode)
    transform = None
    if augment:
        from data.augmentations import AdvancedAugmentations
        transform = AdvancedAugmentations((img_size, img_size))
    cls = _MaskDataset if input_mode == "mask" else ImageDataset
    ds = cls(pairs, transform=transform, channel_first=True, threshold=0.0,
             return_metadata=substrate_mode != "none")
    ds.rows = rows
    g = None
    if seed is not None:
        g = torch.Generator(); g.manual_seed(seed)
    if num_workers > 0:   # workers run the augmentations so the GPU is not left waiting
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g, num_workers=num_workers,
                          persistent_workers=True, pin_memory=torch.cuda.is_available(),
                          worker_init_fn=_worker_init)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g, num_workers=0)
