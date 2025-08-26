# core/train/dataset.py
import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from torch.utils.data import Dataset
from PIL import Image


class ImageCaptionDataset(Dataset):
    """Dataset for image captioning and T2I training"""

    def __init__(
        self,
        data_dir: str,
        metadata_file: Optional[str] = None,
        image_size: int = 768,
        split: str = "train",
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.split = split

        # Load metadata
        if metadata_file and Path(metadata_file).exists():
            if metadata_file.endswith(".parquet"):
                self.metadata = pd.read_parquet(metadata_file)
            else:
                self.metadata = pd.read_csv(metadata_file)

            # Filter by split
            if "split" in self.metadata.columns:
                self.metadata = self.metadata[self.metadata["split"] == split]
        else:
            # Scan directory for images
            self.metadata = self._scan_directory()

    def _scan_directory(self) -> pd.DataFrame:
        """Scan directory for images and create basic metadata"""
        images = []
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            images.extend(list(self.data_dir.glob(f"*{ext}")))
            images.extend(list(self.data_dir.glob(f"*{ext.upper()}")))

        data = []
        for img_path in images:
            data.append(
                {
                    "image_path": str(img_path),
                    "caption": img_path.stem,  # Use filename as caption
                    "split": "train",
                }
            )

        return pd.DataFrame(data)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # Load image
        image_path = row["image_path"]
        if not Path(image_path).is_absolute():
            image_path = self.data_dir / image_path

        try:
            image = Image.open(image_path).convert("RGB")
            # Resize to target size
            image = image.resize((self.image_size, self.image_size))
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Create a dummy image
            image = Image.new(
                "RGB", (self.image_size, self.image_size), color=(128, 128, 128)
            )

        # Get caption
        caption = row.get("caption", "")

        return {"image": image, "caption": caption, "image_path": str(image_path)}
