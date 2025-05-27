#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from datasets.cub import CUBDataset
from models.clip_lora import get_clip_with_lora

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LoRA adapters on CUB-200-2011")
    parser.add_argument("--cfg", type=str, required=True,
                        help="Path to YAML config (e.g. configs/cub.yaml)")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.cfg)

    # Paths
    cub_root = Path(cfg.data.cub_root)
    train_split = cub_root / cfg.data.train_split

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model + processor with LoRA
    model, processor = get_clip_with_lora(
        clip_model_name=cfg.model.clip_model_name,
        local_model_path=getattr(cfg.model, 'local_clip_path', None),
        lora_rank=cfg.model.get('lora_rank', 8),
        lora_alpha=cfg.model.get('lora_alpha', 16),
        lora_dropout=cfg.model.get('lora_dropout', 0.05)
    )
    model.train().to(device)

    # Dataset & DataLoader
    train_ds = CUBDataset(
        cub_root=cub_root,
        split_file=train_split,
        processor=processor
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.eval.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)

    # Training loop
    for epoch in range(cfg.train.epochs):
        for images, labels in train_dl:
            images = images.to(device)
            # Prepare text prompts
            text_inputs = [cfg.train_prompts[0].format(label=str(lbl)) for lbl in labels]
            batch = processor(text=text_inputs,
                              return_tensors="pt",
                              padding=True)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward + loss
            outputs = model(pixel_values=images,
                            input_ids=input_ids,
                            attention_mask=attention_mask)
            logits = outputs.logits_per_image
            targets = torch.arange(images.size(0), device=device)
            loss = torch.nn.functional.cross_entropy(logits, targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{cfg.train.epochs} — Loss: {loss.item():.4f}")

    # Save LoRA-adapted model
    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    print(f"Saved LoRA adapters to {output_dir}")

if __name__ == "__main__":
    main()
