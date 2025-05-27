from pathlib import Path
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image


def default_transform(image: Image.Image, image_size: int = 224) -> torch.Tensor:
    """
    Basic transform: resize to square and center-crop, convert to tensor.
    This can be replaced by CLIPProcessor transforms in the pipeline.
    """
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711)),
    ])
    return transform(image)


class CUBDataset(Dataset):
    """
    PyTorch dataset for CUB-200-2011. Reads a split file with lines:
      <relative_image_path> <class_idx>
    and returns (image_tensor, label).
    """
    def __init__(
        self,
        cub_root: Path,
        split_file: Path,
        processor: Optional[object] = None,
        image_size: int = 224,
    ):
        """
        Args:
          cub_root: Path to the CUB_200_2011 directory (contains 'images/')
          split_file: Path to a splits/*.txt file
          processor: optional CLIPProcessor for image preprocessing
          image_size: size for default transforms if processor is None
        """
        self.cub_root = Path(cub_root)
        self.split_file = Path(split_file)
        self.processor = processor
        self.image_size = image_size

        # Read split entries
        if not self.split_file.is_file():
            raise FileNotFoundError(f"Split file not found: {self.split_file}")
        self.samples: List[Tuple[Path, int]] = []
        with open(self.split_file, 'r') as f:
            for line in f:
                rel_path, cls = line.strip().split()
                img_path = self.cub_root / rel_path
                self.samples.append((img_path, int(cls)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.processor is not None:
            # Use CLIPProcessor for image transforms
            inputs = self.processor(images=image, return_tensors='pt')
            # pixel_values shape: (1, 3, H, W)
            pixel_values = inputs.pixel_values.squeeze(0)
        else:
            pixel_values = default_transform(image, self.image_size)

        text = str(label)
        return {
            "images": pixel_values,
            "texts": text
        }
