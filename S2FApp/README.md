---
title: Shape2Force
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

1. Choose **Model type**: Single cell or Spheroid
2. Select a **Checkpoint** from `ckp/`
3. For single-cell: pick **Substrate** (e.g. fibroblasts_PDMS)
4. Upload an image or pick from `samples/`
5. Click **Run prediction**

Output: heatmap, cell force (sum), and basic stats.

## Full Project

For training, evaluation, and notebooks, see the main [Shape2Force repository](https://github.com/Angione-Lab/Shape2Force).
