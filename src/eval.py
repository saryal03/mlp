#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import faiss
from omegaconf import OmegaConf

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval performance using FAISS and precomputed embeddings"
    )
    parser.add_argument(
        "--cfg", type=str, required=True,
        help="Path to YAML config file (e.g. configs/cub.yaml)"
    )
    parser.add_argument(
        "--split", type=str, default="test",
        choices=["train", "val", "test"],
        help="Which split to evaluate (queries)"
    )
    parser.add_argument(
        "--features", type=str, default="features",
        help="Directory where embeddings (.npy) are stored"
    )
    parser.add_argument(
        "--index-file", type=str, default=None,
        help="Path to a FAISS index file; if unset, builds from train embeddings"
    )
    parser.add_argument(
        "--top-k", type=str, default="1,5,10",
        help="Comma-separated list of K values for Recall@K"
    )
    return parser.parse_args()


def compute_recall(I: np.ndarray, test_labels: np.ndarray, train_labels: np.ndarray, k: int) -> float:
    """
    Computes Recall@k: fraction of queries for which at least one of the top-k neighbors matches the query label.
    """
    # I shape: (num_queries, k_max)
    num = test_labels.shape[0]
    correct = 0
    for i in range(num):
        # get labels of top-k retrieved items
        retrieved = train_labels[I[i, :k]]
        if test_labels[i] in retrieved:
            correct += 1
    return correct / num


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.cfg)

    feat_dir = Path(args.features)
    split = args.split
    topk_list = [int(x) for x in args.top_k.split(",") if x.strip()]

    # Load embeddings and labels
    test_embeds = np.load(feat_dir / f"{split}_embeds.npy")
    test_labels = np.load(feat_dir / f"{split}_labels.npy")
    train_embeds = np.load(feat_dir / "train_embeds.npy")
    train_labels = np.load(feat_dir / "train_labels.npy")

    # Ensure float32
    test_embeds = test_embeds.astype(np.float32)
    train_embeds = train_embeds.astype(np.float32)

    # Load or build FAISS index
    if args.index_file and Path(args.index_file).is_file():
        index = faiss.read_index(str(args.index_file))
    else:
        # Build IVF-PQ index using cfg.eval.faiss_nlist (or default 100)
        nlist = getattr(cfg.eval, 'faiss_nlist', 100)
        quantizer = faiss.IndexFlatIP(train_embeds.shape[1])
        index = faiss.IndexIVFPQ(quantizer,
                                 train_embeds.shape[1],
                                 nlist,
                                 16,
                                 8)
        print(f"Training FAISS index with {train_embeds.shape[0]} vectors, nlist={nlist}")
        index.train(train_embeds)
        index.add(train_embeds)

    # Perform search
    k_max = max(topk_list)
    D, I = index.search(test_embeds, k_max)  # distances, indices

    # Compute and print recalls
    recalls = {k: compute_recall(I, test_labels, train_labels, k) for k in topk_list}
    for k, r in recalls.items():
        print(f"Recall@{k}: {r:.4f}")

if __name__ == "__main__":
    main()
