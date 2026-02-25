---
tags:
  - biology
  - microscopy
  - traction-force
  - pytorch
  - pix2pix
  - gan
license: mit
---

# Shape2Force (S2F)

Predict force maps from bright-field microscopy images of single-cell or spheroid. 

If you find this software useful, please cite:

Lautaro Baro#, Kaveh Shahhosseini#, Amparo Andrés-Bordería, Claudio Angione*, and Maria Angeles Juanes*. **"Shape-to-force (S2F): Predicting Cell Traction Forces from LabelFree Imaging"**, 2026.



---

## Ways to Use S2F

### 1. Web App (local)

Run the Streamlit GUI from `S2FApp/`:
```bash
git clone https://github.com/Angione-Lab/Shape2Force.git
cd Shape2Force/S2FApp
pip install -r requirements.txt
streamlit run app.py
```
1. Choose Model type: Single cell or Spheroid
2. Place checkpoints (`.pth`) in `S2FApp/ckp/` for local use.
3. Select a Checkpoint from `ckp/`
4. For single-cell: pick Substrate (e.g. fibroblasts_PDMS)
5. Upload an image or pick from `samples/`
6. Click Run prediction

---
### 2. Web App Online

Use the [online app](https://huggingface.co/spaces/Angione-Lab/Shape2force) on Hugging Face. 

<p align="center">
  <img src="./S2FApp/res/ss.png" width="450" alt="Shape2Force Web App" />
</p>

---
### 3. Jupyter Notebooks

For interactive usage and custom analysis, use the notebooks in `notebooks/`:

- **`notebooks/Singlecell_inference.ipynb`** – Load a folder of brightfield images, run single-cell predictions, plot samples, and save all predictions with metrics.
- **`notebooks/Singlecell_evaluation.ipynb`** – Evaluate single-cell model on a dataset with ground truth; compute metrics and plot predictions.
- **`notebooks/Spheroid_inference.ipynb`** – Run spheroid predictions on brightfield images, plot samples, and save predictions.
- **`notebooks/Spheroid_evaluation.ipynb`** – Evaluate spheroid model on as dataset with ground truth; compute metrics and plot predictions.

Once cloned, open a notebook in Jupyter and adjust the configuration cell (paths, model type, substrate).

---

### 4. Training & Fine-Tuning

**Dataset layout:** A folder with `train/` and `test/` subfolders. Each subfolder has:
- `BF_001.tif` (bright-field image)
- `*_gray.jpg` (force map / heatmap)
- Optional `.txt` (cell_area, sum_force)

**Single-cell:**
```bash
python -m training.train \
  --data path/to/dataset \
  --model single_cell \
  --epochs 100 \
  --substrate fibroblasts_PDMS
```

**Spheroid:**
```bash
python -m training.train \
  --data path/to/dataset \
  --model spheroid \
  --epochs 100
```

**Resume / fine-tune from checkpoint:**
```bash
python -m training.train \
  --data path/to/dataset \
  --model single_cell \
  --resume ckp/last_checkpoint.pth \
  --epochs 150
```

---

