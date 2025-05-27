#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

# Assumes project root is on PYTHONPATH or use relative imports
from datasets.cub import CUBDataset
from models.clip_lora import get_clip_with_lora


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract CLIP embeddings for a given CUB split"
    )
    parser.add_argument(
        "--cfg", type=str, required=True,
        help="Path to YAML config file (e.g. configs/cub.yaml)"
    )
    parser.add_argument(
        "--split", type=str, required=True,
        choices=["train", "val", "test"],
        help="Which data split to process"
    )
    parser.add_argument(
        "--output", type=str, default="features",
        help="Directory where to save embedding .npy files"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.cfg)

    cub_root = Path(cfg.data.cub_root)
    split_paths = {
        "train": cfg.data.train_split,
        "val":   cfg.data.val_split,
        "test":  cfg.data.test_split,
    }
    split_file = cub_root / split_paths[args.split]
    if not split_file.is_file():
        raise FileNotFoundError(f"Split not found: {split_file}")

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and processor (LoRA if configured)
    model, processor = get_clip_with_lora(
        cfg.model.clip_model_name,
        local_model_path=getattr(cfg.model, 'local_clip_path', None),
        lora_rank=cfg.model.get('lora_rank', 8),
        lora_alpha=cfg.model.get('lora_alpha', 16),
        lora_dropout=cfg.model.get('lora_dropout', 0.05)
    )
    model.eval().to(device)

    # Prepare dataset & loader
    ds = CUBDataset(
        cub_root=cub_root,
        split_file=split_file,
        processor=processor
    )
    dl = DataLoader(
        ds,
        batch_size=cfg.eval.batch_size,
        shuffle=False,
        num_workers=cfg.eval.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Extract embeddings
    all_embeds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dl:
            images = images.to(device)
            feats = model.get_image_features(images)
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)
            all_embeds.append(feats.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_embeds = np.concatenate(all_embeds, axis=0)
    all_labels = np.array(all_labels)

    # Save to disk
    np.save(os.path.join(args.output, f"{args.split}_embeds.npy"), all_embeds)
    np.save(os.path.join(args.output, f"{args.split}_labels.npy"), all_labels)
    print(f"Saved {all_embeds.shape[0]} embeddings to {args.output}")


if __name__ == "__main__":
    main()
