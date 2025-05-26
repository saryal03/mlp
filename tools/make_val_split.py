#!/usr/bin/env python3
import random
from pathlib import Path

# 1️⃣ Point to your CUB root
cub_root = Path("/Users/sa/Documents/SCU/MLProject/data/CUB_200_2011")
splits_dir = cub_root / "splits"
train_file = splits_dir / "train.txt"

# 2️⃣ Read all lines, shuffle
lines = train_file.read_text().splitlines()
random.shuffle(lines)

# 3️⃣ Split 10% for val, 90% back to train
n_total = len(lines)
n_val   = int(0.1 * n_total)
val_lines   = lines[:n_val]
new_train   = lines[n_val:]

# 4️⃣ Write out
(splits_dir / "val.txt").write_text("\n".join(val_lines) + "\n")
train_file.write_text("\n".join(new_train) + "\n")

print(f"Created val.txt ({n_val} samples) and updated train.txt ({len(new_train)} samples).")
