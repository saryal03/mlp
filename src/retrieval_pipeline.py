#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import faiss

from transformers import CLIPModel, CLIPProcessor
from transformers.models.clip.modeling_clip import CLIPTextTransformer
from peft import LoraConfig, get_peft_model
from tqdm import tqdm 
from datasets.cub import CUBDataset

# ── Patch CLIPTextTransformer to drop any unexpected inputs_embeds ──────────────
_original_clip_text_forward = CLIPTextTransformer.forward
def _patched_clip_text_forward(self, *args, **kwargs):
    # quietly remove inputs_embeds if PEFT tries to pass it
    kwargs.pop("inputs_embeds", None)
    return _original_clip_text_forward(self, *args, **kwargs)

CLIPTextTransformer.forward = _patched_clip_text_forward
# ───────────────────────────────────────────────────────────────────────────────


def get_clip_with_lora(cfg):
    """
    Load CLIP, wrap only the text encoder in LoRA, and return (model, processor).
    """
    # 1) Load base CLIP model + processor
    model = CLIPModel.from_pretrained(cfg.model.clip_model_name)
    processor = CLIPProcessor.from_pretrained(
        cfg.model.clip_model_name,
        use_fast=True)

    # 2) Configure LoRA
    peft_config = LoraConfig(
        task_type="FEATURE_EXTRACTION",  # ← changed here
        inference_mode=False,
        r=cfg.train.lora_r,
        lora_alpha=cfg.train.lora_alpha,
        lora_dropout=cfg.train.lora_dropout,
        target_modules=["q_proj", "v_proj"],
    )

    # 3) Wrap only the text encoder so that CLIPModel.forward()
    #    doesn’t get an unexpected `inputs_embeds` argument.
    model.text_model = get_peft_model(model.text_model, peft_config)

    return model, processor


def train_clip_lora(model, processor, train_dl, cfg):
    """
    Standard CLIP‐style contrastive training loop, except
    model.text_model carries the LoRA adapters.
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("Using Apple GPU via MPS")
    else:
        device = torch.device("cpu")
        print("Falling back to CPU")

    model.to(device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    model.train()

    for epoch in range(cfg.train.epochs):
        # tqdm progress bar for training
        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{cfg.train.epochs}"):
            images = batch["images"].to(device)
            texts = batch["texts"]

            encoding = processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)

            outputs = model(
                pixel_values=images,
                input_ids=encoding.input_ids,
                attention_mask=encoding.attention_mask,
            )

            # contrastive loss
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds

            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

            logit_scale = model.logit_scale.exp()
            logits_per_image = logit_scale * image_embeds @ text_embeds.t()
            logits_per_text  = logits_per_image.t()
            labels = torch.arange(logits_per_image.size(0), device=device)

            loss_i = torch.nn.functional.cross_entropy(logits_per_image, labels)
            loss_t = torch.nn.functional.cross_entropy(logits_per_text,  labels)
            loss   = (loss_i + loss_t) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{cfg.train.epochs} — loss: {loss.item():.4f}")

    # save LoRA‐augmented weights
    os.makedirs(cfg.train.output_dir, exist_ok=True)
    model.save_pretrained(cfg.train.output_dir)
    print(f"\n✅ Training complete! LoRA weights saved to: {cfg.train.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train CLIP + LoRA on CUB")
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        required=True,
        help="Path to your cub.yaml"
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    # build dataset & loader (pass processor later)
    _, processor = get_clip_with_lora(cfg)
    train_ds = CUBDataset(
        cub_root=cfg.data.cub_root,
        split_file=cfg.data.train_split,
        processor=processor
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.eval.batch_size,
        num_workers=cfg.eval.num_workers,
        shuffle=True
    )

    model, _ = get_clip_with_lora(cfg)
    train_clip_lora(model, processor, train_dl, cfg)


if __name__ == "__main__":
    main()