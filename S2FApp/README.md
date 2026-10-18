---
title: Shape2Force
short_description: Force map prediction from bright-field cell images
emoji: 🦠
colorFrom: indigo
colorTo: blue
tags:
- cell-mechanobiology
- microscopy
- image-to-image
- pytorch
license: cc-by-4.0
sdk: docker
app_port: 8501
---

# Shape2Force (S2F) App

Predict force maps from bright-field microscopy images using deep learning. 

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Checkpoints are downloaded automatically from the [Shape2Force model repo](https://huggingface.co/Angione-Lab/Shape2Force) when running in Docker. For local use, place `.pth` files in `ckp/`.

## Usage

1. Choose **Model type**: Single cell or Spheroid.
2. Select a **Checkpoint** from `ckp/`. The architecture is read from the checkpoint. For spheroids the default is
   `ckp_spheroid_FN.pth`, a smaller supervised model (11.8 M parameters) with smoother maps and similar accuracy;
   `ckp_spheroid_GN.pth` is the adversarial S2F model (77 M parameters).
3. Single cell: keep the **Substrate setting** at *Reference (6 kPa)*. The released model is most accurate with this
   setting for every image, including cells on 1 or 10 kPa gels; presets and manual values are for exploration.
4. Enter the **Image pixel size** (µm/px) of your bright-field image, for the scale bar, area and total force
   (0.325 µm/px for the single-cell training images).
5. Upload an image or pick one from `samples/`, then click **Run prediction**.

## Output and units

- The model predicts a normalised map in [0, 1] that converts to **traction stress** as
  T = 500 Pa x value (single cells) or 50 Pa x value (spheroids), the fixed colour limits of the TFM ground-truth
  maps used for training. All maps carry a colourbar in Pa and a µm scale bar.
- Metrics: mean and maximum traction stress (Pa), area (µm²) and total force (nN = Σ stress x pixel area). By default
  (*Measure force region only*) they are computed inside the region of high predicted stress (red contour), estimated
  from the predicted map, so that the background does not dilute them; switch it off for the whole image.
  S2F predicts magnitude only, not direction.
- **Display range** narrows the colour window (e.g. 0–250 Pa); stresses above it saturate. Display only.
- **Downloads**: map PNG (scale bar + colourbar), CSV with metrics and parameters, and a PDF report that records the
  input file, model, checkpoint SHA-256, substrate setting, pixel size, display and force-region settings and app
  version. Batch mode gives a PDF (summary table + one page per image), a CSV, and a ZIP with, per image, the PNG and a
  TIFF whose pixel values are traction stress in Pa (for measurements in Fiji/ImageJ). Set `S2F_APP_COMMIT` to add
  the git commit.

## Full Project

For training, evaluation, and notebooks, see the main [Shape2Force repository](https://github.com/Angione-Lab/Shape2Force).
