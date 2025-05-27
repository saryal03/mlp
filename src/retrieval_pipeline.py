import argparse
import os
from pathlib import Path
from typing import List, Tuple
from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel

try:
    from peft import LoraConfig, get_peft_model
    import peft
except ImportError:
    peft = None

try:
    import faiss
except ImportError:
    faiss = None

###############################
# CONFIGURATION
###############################
class Config:
    # HuggingFace model identifier and local path
    clip_model_name: str = "openai/clip-vit-base-patch16"
    local_clip_path: Path = Path(__file__).resolve().parent / "CLIP" / "clip"

    # Prompt templates
    train_prompts: List[str] = [
        "a photo of a {label}",
        "a close-up photo of a {label}",
        "a detailed image of a {label}",
    ]

    # LoRA settings
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # Dataset location (override via CLI)
    cub_root: Path = Path("./data/CUB_200_2011")

    # Training settings
    batch_size: int = 64
    epochs: int = 1
    lr: float = 1e-4
    num_workers: int = 4

    # Retrieval settings
    faiss_nlist: int = 100
    top_k: Tuple[int, ...] = (1, 5, 10)


def parse_args():
    parser = argparse.ArgumentParser(description="CLIP retrieval pipeline with optional LoRA.")
    parser.add_argument("--cub_root", type=Path, default=Config.cub_root,
                        help="Path to CUB_200_2011 root (contains images/ and splits/)")
    parser.add_argument("--batch_size", type=int, default=Config.batch_size)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--num_workers", type=int, default=Config.num_workers)
    return parser.parse_args()

###############################
# DATASET
###############################
class ImageLabelDataset(torch.utils.data.Dataset):
    """Minimal image-path/label dataset; expects list[(path,label)]."""

    def __init__(self, samples: List[Tuple[str, int]], processor: CLIPProcessor):
        self.samples = samples
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        return pixel_values, label


def build_samples_list(root: Path, split_file: str) -> List[Tuple[str, int]]:
    """Reads split info and returns a list of (image_path, class_idx)."""
    split_path = root / split_file
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    samples: List[Tuple[str, int]] = []
    with open(split_path, "r") as f:
        for line in f:
            img_rel_path, cls = line.strip().split()
            samples.append((str(root / img_rel_path), int(cls)))
    return samples

###############################
# MODEL SETUP WITH OPTIONAL LoRA
###############################
def get_clip_with_lora(cfg: Config) -> Tuple[CLIPModel, CLIPProcessor]:
    model_path = cfg.local_clip_path
    # Load locally if available, else download from HF
    if model_path.is_dir():
        model = CLIPModel.from_pretrained(str(model_path), local_files_only=True)
        processor = CLIPProcessor.from_pretrained(str(model_path), local_files_only=True)
    else:
        print(f"Local model not found at {model_path}, downloading from Hugging Face...")
        model = CLIPModel.from_pretrained(cfg.clip_model_name)
        processor = CLIPProcessor.from_pretrained(cfg.clip_model_name)

    # Apply LoRA if PEFT is installed
    if peft is not None:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type=peft.TaskType.FEATURE_EXTRACTION,
        )
        model = get_peft_model(model, lora_config)
    else:
        print("⚠️ PEFT not installed; running without LoRA adapters.")

    return model, processor

###############################
# TRAINING LOOP
###############################
def train_clip_lora(model: CLIPModel,
                    processor: CLIPProcessor,
                    train_dl: DataLoader,
                    cfg: Config) -> None:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(cfg.epochs):
        for images, labels in train_dl:
            images = images.to(device)
            # Format text prompts with label
            text_inputs = [cfg.train_prompts[0].format(label=str(label)) for label in labels]

            # Tokenize and move tensors explicitly
            batch = processor(text=text_inputs, return_tensors="pt", padding=True)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass
            outputs = model(
                pixel_values=images,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits_per_image = outputs.logits_per_image
            targets = torch.arange(images.size(0), device=device)
            loss = nn.functional.cross_entropy(logits_per_image, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{cfg.epochs} — Loss: {loss.item():.4f}")

###############################
# EMBEDDING & INDEXING
###############################
def extract_embeddings(model: CLIPModel,
                       dataloader: DataLoader) -> Tuple[torch.Tensor, List[int]]:
    model.eval()
    device = next(model.parameters()).device
    all_embeds, all_labels = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            embeds = model.get_image_features(images)
            embeds = nn.functional.normalize(embeds, p=2, dim=1)
            all_embeds.append(embeds.cpu())
            all_labels.extend(labels)
    return torch.cat(all_embeds, dim=0), all_labels


def build_faiss_index(vectors: torch.Tensor, cfg: Config):
    if faiss is None:
        raise ImportError("faiss not installed; cannot build index.")
    dim = vectors.shape[1]
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, cfg.faiss_nlist, 16, 8)
    index.train(vectors.numpy())
    index.add(vectors.numpy())
    return index

###############################
# EVALUATION
###############################
def recall_at_k(results: List[List[int]], labels: List[int], k: int) -> float:
    correct = sum(labels[i] in results[i][:k] for i in range(len(labels)))
    return correct / len(labels)

###############################
# MAIN
###############################
def main():
    args = parse_args()
    cfg = Config()
    # Override config from CLI
    cfg.cub_root = args.cub_root
    cfg.batch_size = args.batch_size
    cfg.epochs = args.epochs
    cfg.lr = args.lr
    cfg.num_workers = args.num_workers

    # 1) Load model & processor
    model, processor = get_clip_with_lora(cfg)

    # 2) Prepare data loader
    train_samples = build_samples_list(cfg.cub_root, "splits/train.txt")
    train_ds = ImageLabelDataset(train_samples, processor)
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # 3) Fine-tune with LoRA (or base CLIP)
    train_clip_lora(model, processor, train_dl, cfg)
    if peft is not None:
        model.save_pretrained("./checkpoints/clip_lora")

    # 4) Extract embeddings and build FAISS index
    vectors, labels = extract_embeddings(model, train_dl)
    index = build_faiss_index(vectors, cfg)

    # 5) Evaluate recall on training set (demo)
    D, I = index.search(vectors.numpy(), max(cfg.top_k))
    recalls = {k: recall_at_k(I.tolist(), labels, k) for k in cfg.top_k}
    print("Recall:", recalls)

if __name__ == "__main__":
    main()