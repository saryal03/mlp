#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import numpy as np
import faiss
from omegaconf import OmegaConf

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build FAISS IVF-PQ index from gallery embeddings"
    )
    parser.add_argument(
        "--cfg", type=str, required=True,
        help="Path to YAML config file (e.g. configs/cub.yaml)"
    )
    parser.add_argument(
        "--split", type=str, default="train",
        choices=["train", "val", "test"],
        help="Which split to build gallery from (default: train)"
    )
    parser.add_argument(
        "--emb_dir", type=str, default="features",
        help="Directory where embeddings are saved"
    )
    parser.add_argument(
        "--index_file", type=str, default="features/faiss_index.ivfpq",
        help="Output path for saved FAISS index"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.cfg)

    emb_dir = Path(args.emb_dir)
    emb_path = emb_dir / f"{args.split}_embeds.npy"
    if not emb_path.is_file():
        raise FileNotFoundError(f"Embeddings file not found: {emb_path}")

    # Load gallery embeddings
    gallery = np.load(str(emb_path))
    print(f"Loaded gallery embeddings: {gallery.shape}")

    # Determine FAISS parameters
    nlist = getattr(cfg, 'index', {}).get('faiss_nlist', None)
    if nlist is None:
        # Try model section
        nlist = cfg.get('model', {}).get('faiss_nlist', 100)
    quantizer = faiss.IndexFlatIP(gallery.shape[1])
    index = faiss.IndexIVFPQ(quantizer, gallery.shape[1], int(nlist), 16, 8)

    # Train and add embeddings
    print("Training FAISS IVFPQ index...")
    index.train(gallery)
    index.add(gallery)

    # Write to disk
    os.makedirs(Path(args.index_file).parent, exist_ok=True)
    faiss.write_index(index, args.index_file)
    print(f"FAISS index saved to {args.index_file}")

if __name__ == "__main__":
    main()
