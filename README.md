# Shape2Force (S2F)

Predict force maps from bright-field microscopy images of single-cell or spheroid using deep learning. 

**Web App:** The app is published to [Hugging Face Spaces](https://huggingface.co/spaces/kaveh/Shape2force). To work on it locally: `git clone git@hf.co:spaces/kaveh/Shape2force S2FApp`

---

## Quick Start

**Web app (local):**
```bash
cd S2FApp
pip install -r requirements.txt
streamlit run app.py
```

Or use the [online app](https://huggingface.co/spaces/kaveh/Shape2force) on Hugging Face. Place checkpoints (`.pth`) in `S2FApp/ckp/` for __local use__; the Space downloads them automatically.

---

## Ways to Use S2F

### 1. Web App

Run the Streamlit GUI from `S2FApp/`:

```bash
cd S2FApp && streamlit run app.py
```

1. Choose **Model type**: Single cell or Spheroid
2. Select a **Checkpoint** from `ckp/`
3. For single-cell: pick **Substrate** (e.g. fibroblasts_PDMS)
4. Upload an image or pick from `samples/`
5. Click **Run prediction**

Output: heatmap, cell force (sum), and basic stats.

----

### 2. Jupyter Notebook

For interactive usage and custom analysis, you may use the notebook:

- **`notebooks/evaluate_model.ipynb`** – Load data, run evaluation, plot predictions, and save per-sample metrics.

Once cloned the repo. open the notebook in Jupyter and adjust the configuration cell (paths, model type, substrate).

---

### 3. Training & Fine-Tuning

**Dataset layout:** A folder with `train/` and `test/` subfolders. Each subfolder has:
- `BF_001.tif` (bright-field image)
- `*_gray.jpg` (force map / heatmap)
- Optional `.txt` (cell_area, sum_force)

**Single-cell:**
```bash
python -m training.train --data path/to/dataset --model single_cell --epochs 100 --substrate fibroblasts_PDMS
```

**Spheroid:**
```bash
python -m training.train --data path/to/dataset --model spheroid --epochs 100
```

**Resume / fine-tune from checkpoint:**
```bash
python -m training.train --data path/to/dataset --model single_cell --resume ckp/last_checkpoint.pth --epochs 150
```

---

