# CLIP+LoRA Image Retrieval on CUB-200-2011

## Overview

This project demonstrates how to fine-tune OpenAI's CLIP model with LoRA adapters for efficient image–text retrieval on the CUB-200-2011 bird dataset, and how to perform fast inference with caching and progress bars.

Authors: Harsh, Justin, Sudip Machine Learning, CSEN 240 SCU Spring Quarter 2025
Key components:

* **Training pipeline** (`src/retrieval_pipeline.py`): loads CLIP, wraps the text encoder with LoRA, and trains on CUB splits.
* **Inference script** (`scripts/inference.py`): performs zero-shot text→image search using the fine-tuned model, with optional caching of gallery embeddings.
* **Configuration** (`configs/cub.yaml`): hyperparameters and data paths.
* **Caching** (`cache/`): stores precomputed FAISS index and embeddings for sub-second inference startup.

## Project Structure

```
MLProject/
├── .gitignore           # Ignore venv, data, outputs, cache, etc.
├── README.md            # This file
├── requirements.txt     # Python dependencies
├── configs/
│   └── cub.yaml         # Dataset paths & training settings
├── src/
│   └── retrieval_pipeline.py  # Training script
├── scripts/
│   └── inference.py     # Retrieval & demo script
├── data/
│   └── CUB_200_2011/     # Downloaded CUB images & splits
├── outputs/
│   └── lora/            # Saved LoRA adapter weights
└── cache/
    ├── gallery.index    # FAISS index (auto-generated)
    ├── gallery_embs.npy # Gallery embeddings (auto-generated)
    └── image_paths.pkl  # List of filepaths (auto-generated)
```

## Installation

1. **Clone the repo** and `cd` into it:

   ```bash
   git clone <r........> MLProject
   cd MLProject
   ```

2. **Create & activate** a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Configuration

Edit `configs/cub.yaml` to point at your local CUB dataset and adjust hyperparameters:

```yaml
data:
  cub_root: "/path/to/CUB_200_2011"
  train_split: "splits/train.txt"
  val_split:   "splits/val.txt"
  test_split:  "splits/test.txt"

eval:
  batch_size: 64
  num_workers: 4

model:
  clip_model_name: "openai/clip-vit-base-patch16"

train:
  lr: 1e-4
  epochs: 1
  lora_r: 8
  lora_alpha: 32
  lora_dropout: 0.05
  output_dir: "outputs/lora"

index:
  faiss_nlist: 100
  top_k: [1, 5, 10]
```

## Training

Run the training pipeline to fine-tune CLIP with LoRA adapters:

```bash
.venv/bin/python src/retrieval_pipeline.py \
  --config configs/cub.yaml
```

You will see progress bars and loss per epoch.
After completion, the adapter weights are saved under `outputs/lora`.

## Inference

Perform text→image retrieval with your fine-tuned model:

```bash
.venv/bin/python scripts/inference.py
```

1. Enter your text query at the prompt.
2. The script will load (or build) a FAISS index of the gallery images, embed your query, and display the top-\`5\` results in a row, along with file paths and similarity scores.

## Caching for Speed

* On the **first run**, `scripts/inference.py` computes and caches:

  * `cache/gallery.index` (FAISS index)
  * `cache/gallery_embs.npy` (embeddings)
  * `cache/image_paths.pkl` (file list)

* On **subsequent runs**, it loads these files directly, skipping the costly re-embedding step for near-instant startup.

To force a rebuild (e.g. after adding images), delete the contents of `cache/` and rerun the script.

## .gitignore

 `.gitignore` excludes:

```
.venv/
__pycache__/
data/
outputs/
cache/
*.bin
*.npy
*.pkl
```

## License

Feel free to adapt and redistribute.
