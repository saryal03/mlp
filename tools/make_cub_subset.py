#!/usr/bin/env python3
import random
from pathlib import Path

# 1. Point to your CUB images folder
cub_root = Path("/Users/sa/Documents/SCU/MLProject/data/CUB_200_2011")
images_dir = cub_root / "images"

# 2. Gather all (image_path, class_idx) pairs
all_samples = []
for class_folder in sorted(images_dir.iterdir()):
    if not class_folder.is_dir(): continue
    class_idx = int(class_folder.name.split(".")[0]) - 1  # zero‑based
    for img in class_folder.glob("*.jpg"):
        # store relative path as in the official splits
        rel = img.relative_to(cub_root)
        all_samples.append((str(rel), class_idx))

# 3. Randomly sample 3,000
subset = random.sample(all_samples, k=3000)

# 4. Write train.txt
splits_dir = cub_root / "splits"
splits_dir.mkdir(exist_ok=True)
with open(splits_dir / "train.txt", "w") as f:
    for rel_path, cls in subset:
        f.write(f"{rel_path} {cls}\n")

print(f"Wrote {len(subset)} entries to {splits_dir/'train.txt'}")
