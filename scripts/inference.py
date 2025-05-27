#!/usr/bin/env python3
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"  # macOS OpenMP workaround
import torch
from transformers import CLIPModel, CLIPProcessor
import faiss
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# ── Configuration ─────────────────────────────────────────────────────────
IMAGE_FOLDER = "data/CUB_200_2011/images"
MODEL_DIR    = "outputs/lora"
BASE_CLIP    = "openai/clip-vit-base-patch16"
TOP_K        = 5

CACHE_DIR    = "cache"
INDEX_PATH   = os.path.join(CACHE_DIR, "gallery.index")
EMBS_PATH    = os.path.join(CACHE_DIR, "gallery_embs.npy")
PATHS_PATH   = os.path.join(CACHE_DIR, "image_paths.pkl")


# ── Helpers ────────────────────────────────────────────────────────────────
def load_index_and_paths():
    index = faiss.read_index(INDEX_PATH)
    with open(PATHS_PATH, "rb") as f:
        image_paths = pickle.load(f)
    return index, image_paths


def build_and_save_index(model, processor, device, image_folder, batch_size=64):
    # 1) collect image filepaths
    image_paths = []
    for root, _, files in os.walk(image_folder):
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(root, fname))

    # 2) batch-embed images
    all_embs = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=imgs, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        all_embs.append(feats.cpu().numpy())

    # 3) stack and build FAISS index
    all_embs = np.vstack(all_embs).astype("float32")
    dim      = all_embs.shape[1]
    index    = faiss.IndexFlatIP(dim)
    index.add(all_embs)

    # 4) cache to disk
    os.makedirs(CACHE_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    np.save(EMBS_PATH, all_embs)
    with open(PATHS_PATH, "wb") as f:
        pickle.dump(image_paths, f)

    return index, image_paths


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    # ask for user query
    query = input("Enter your text query: ").strip()
    print()

    # select device (MPS on Mac, else CPU)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("🚀 Using Apple GPU via MPS")
    else:
        device = torch.device("cpu")
        print("⚙️  Falling back to CPU")

    # load model & processor
    model     = CLIPModel.from_pretrained(MODEL_DIR).to(device)
    processor = CLIPProcessor.from_pretrained(BASE_CLIP)

    # load or build/cache gallery index
    if os.path.exists(INDEX_PATH) and os.path.exists(PATHS_PATH):
        index, image_paths = load_index_and_paths()
        print("♻️  Loaded gallery index from cache")
    else:
        print("⏳ Building gallery index (first run)…")
        index, image_paths = build_and_save_index(
            model, processor, device, IMAGE_FOLDER
        )
        print("✅ Built gallery index and cached to disk")

    # embed query & search
    print("\nEmbedding query & searching…")
    txt_inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        qv = model.get_text_features(**txt_inputs)
    qv = qv / qv.norm(dim=-1, keepdim=True)

    D, I = index.search(qv.cpu().numpy(), TOP_K)

    # display results in a row
    scores = D[0]
    idxs   = I[0]
    fig, axes = plt.subplots(1, TOP_K, figsize=(TOP_K * 2, 2.5))
    fig.suptitle(f'Query: "{query}"', y=1.05)
    for rank, (ax, idx) in enumerate(zip(axes, idxs), start=1):
        path = image_paths[idx]
        img  = Image.open(path)
        ax.imshow(img)
        ax.set_title(f"Rank {rank}\n{scores[rank-1]:.2f}", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    # print file paths & scores
    print("\nTop-5 results:")
    for rank, idx in enumerate(idxs, start=1):
        print(f"#{rank}: {image_paths[idx]}  (score {scores[rank-1]:.3f})")

    print("\n✅ Process completed.")


if __name__ == "__main__":
    main()