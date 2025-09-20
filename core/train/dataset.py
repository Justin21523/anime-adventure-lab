# core/train/dataset.py

import os
import json
import logging
import hashlib
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset
from transformers import CLIPTokenizer
import albumentations as A
from albumentations.pytorch import ToTensorV2

from ..config import get_config
from ..utils.image import ImageProcessor, get_image_processor
from ..utils.text import TextProcessor, get_text_processor
from ..exceptions import DatasetError, ValidationError
from .config import TrainingConfig, DatasetConfig

logger = logging.getLogger(__name__)


class TrainingDataset(Dataset):
    """Custom dataset for LoRA training with augmentations"""

    def __init__(
        self,
        config: DatasetConfig,
        tokenizer: CLIPTokenizer,
        resolution: int = 512,
        augment: bool = True,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.augment = augment

        # Initialize processors
        self.image_processor = get_image_processor()
        self.text_processor = get_text_processor()

        # Load dataset
        self.dataset = self._load_dataset()
        self.length = len(self.dataset)

        # Setup augmentations
        self.transforms = self._setup_augmentations() if augment else None

        logger.info(f"📚 Dataset loaded: {self.length} samples")

    def _load_dataset(self) -> HFDataset:
        """Load dataset from various sources"""
        try:
            if self.config.type == "imagefolder":
                # Load from image folder structure
                dataset = load_dataset(
                    "imagefolder", data_dir=self.config.path, split=self.config.split
                )

            elif self.config.type == "json":
                # Load from JSON file
                with open(self.config.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                dataset = HFDataset.from_list(data)

            elif self.config.type == "parquet":
                # Load from Parquet file
                df = pd.read_parquet(self.config.path)
                dataset = HFDataset.from_pandas(df)

            elif self.config.type == "huggingface":
                # Load from Hugging Face Hub
                dataset = load_dataset(self.config.path, split=self.config.split)

            else:
                raise DatasetError(f"Unsupported dataset type: {self.config.type}")

            # Validate required columns
            if self.config.caption_column not in dataset.column_names:
                raise DatasetError(
                    f"Caption column '{self.config.caption_column}' not found"
                )

            if self.config.image_column not in dataset.column_names:
                raise DatasetError(
                    f"Image column '{self.config.image_column}' not found"
                )

            return dataset

        except Exception as e:
            raise DatasetError(f"Failed to load dataset: {e}")

    def _setup_augmentations(self) -> A.Compose:
        """Setup image augmentations"""
        augmentations = [
            # Resize to training resolution
            A.Resize(self.resolution, self.resolution, interpolation=1),
            # Color augmentations (mild)
            A.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3
            ),
            # Geometric augmentations (very mild for character consistency)
            A.HorizontalFlip(p=0.5),
            # Noise and blur (very mild)
            A.OneOf(
                [
                    A.GaussNoise(var_limit=(0, 25), p=0.2),
                    A.GaussianBlur(blur_limit=3, p=0.1),
                    A.MotionBlur(blur_limit=3, p=0.1),
                ],
                p=0.2,
            ),
            # Normalization for diffusion models
            A.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]

        return A.Compose(augmentations)

    def _preprocess_caption(self, caption: str) -> str:
        """Preprocess caption text"""
        # Clean and normalize text
        cleaned_caption = self.text_processor.clean_text(
            caption,
            {
                "remove_urls": True,
                "remove_emails": True,
                "convert_chinese": True,
                "remove_special_chars": False,
            },
        )

        # Apply preprocessing rules from config
        preprocessing = self.config.preprocessing

        if preprocessing.get("add_trigger_word"):
            trigger_word = preprocessing["add_trigger_word"]
            if trigger_word not in cleaned_caption:
                cleaned_caption = f"{trigger_word}, {cleaned_caption}"

        if preprocessing.get("max_caption_length"):
            max_length = preprocessing["max_caption_length"]
            cleaned_caption = self.text_processor.truncate_text(
                cleaned_caption, max_length
            )

        return cleaned_caption

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            # Get data item
            item = self.dataset[idx]

            # Load and process image
            image = item[self.config.image_column]
            if isinstance(image, str):
                # Image path
                image_path = Path(image)
                if not image_path.is_absolute():
                    # Relative to dataset path
                    image_path = Path(self.config.path).parent / image
                image = Image.open(image_path).convert("RGB")
            elif not isinstance(image, Image.Image):
                # Convert array to PIL
                image = Image.fromarray(image).convert("RGB")

            # Apply augmentations or simple resize
            if self.transforms:
                # Convert PIL to numpy for albumentations
                image_array = np.array(image)
                augmented = self.transforms(image=image_array)
                pixel_values = augmented["image"]
            else:
                # Simple resize and normalize
                image = image.resize((self.resolution, self.resolution), Image.LANCZOS)
                image_array = np.array(image).astype(np.float32) / 255.0
                # Normalize to [-1, 1]
                image_array = (image_array - 0.5) / 0.5
                pixel_values = torch.from_numpy(image_array).permute(2, 0, 1)

            # Process caption
            caption = item[self.config.caption_column]
            if isinstance(caption, list):
                caption = caption[0] if caption else ""

            caption = self._preprocess_caption(caption)

            # Tokenize caption
            tokenized = self.tokenizer(
                caption,
                padding="max_length",
                max_length=77,  # CLIP max length
                truncation=True,
                return_tensors="pt",
            )

            return {
                "pixel_values": pixel_values,
                "input_ids": tokenized.input_ids.squeeze(0),
                "attention_mask": tokenized.attention_mask.squeeze(0),
                "caption": caption,
            }

        except Exception as e:
            logger.warning(f"⚠️ Failed to process item {idx}: {e}")
            # Return next item to avoid training interruption
            return self.__getitem__((idx + 1) % self.length)


class DatasetProcessor:
    """Dataset processing and analysis utilities"""

    def __init__(self):
        self.config = get_config()
        self.image_processor = get_image_processor()
        self.text_processor = get_text_processor()

    def analyze_dataset(
        self, dataset_path: str, dataset_type: str = "imagefolder"
    ) -> Dict[str, Any]:
        """Analyze dataset and provide statistics"""
        try:
            # Load dataset
            if dataset_type == "imagefolder":
                dataset = load_dataset(
                    "imagefolder", data_dir=dataset_path, split="train"
                )
            elif dataset_type == "json":
                with open(dataset_path, "r") as f:
                    data = json.load(f)
                dataset = HFDataset.from_list(data)
            elif dataset_type == "parquet":
                df = pd.read_parquet(dataset_path)
                dataset = HFDataset.from_pandas(df)
            else:
                raise DatasetError(f"Unsupported dataset type: {dataset_type}")

            # Basic statistics
            stats = {
                "total_samples": len(dataset),
                "columns": dataset.column_names,
                "features": dict(dataset.features),
            }

            # Image analysis (sample-based for performance)
            if "image" in dataset.column_names:
                image_stats = self._analyze_images(
                    dataset, sample_size=min(100, len(dataset))
                )
                stats["images"] = image_stats

            # Text analysis
            if "text" in dataset.column_names:
                text_stats = self._analyze_captions(
                    dataset, sample_size=min(1000, len(dataset))
                )
                stats["captions"] = text_stats

            return stats

        except Exception as e:
            raise DatasetError(f"Dataset analysis failed: {e}")

    def _analyze_images(self, dataset: HFDataset, sample_size: int) -> Dict[str, Any]:
        """Analyze image properties"""
        import random

        sample_indices = random.sample(range(len(dataset)), sample_size)

        widths, heights, formats, sizes = [], [], [], []

        for idx in sample_indices:
            try:
                image = dataset[idx]["image"]
                if isinstance(image, str):
                    image = Image.open(image)

                width, height = image.size
                widths.append(width)
                heights.append(height)
                formats.append(image.format or "Unknown")

                # Estimate file size
                if hasattr(image, "fp") and image.fp:
                    try:
                        image.fp.seek(0, 2)  # Seek to end
                        size = image.fp.tell()
                        sizes.append(size)
                    except:
                        pass

            except Exception as e:
                logger.warning(f"Failed to analyze image {idx}: {e}")

        return {
            "sample_size": len(widths),
            "resolution": {
                "width_range": (min(widths), max(widths)) if widths else (0, 0),
                "height_range": (min(heights), max(heights)) if heights else (0, 0),
                "avg_width": sum(widths) / len(widths) if widths else 0,
                "avg_height": sum(heights) / len(heights) if heights else 0,
            },
            "formats": {fmt: formats.count(fmt) for fmt in set(formats)},
            "avg_file_size_mb": sum(sizes) / len(sizes) / 1024**2 if sizes else 0,
        }

    def _analyze_captions(self, dataset: HFDataset, sample_size: int) -> Dict[str, Any]:
        """Analyze caption text properties"""
        import random

        sample_indices = random.sample(range(len(dataset)), sample_size)

        lengths, languages, word_counts = [], [], []
        all_words = []

        for idx in sample_indices:
            try:
                caption = dataset[idx]["text"]
                if isinstance(caption, list):
                    caption = caption[0] if caption else ""

                # Basic stats
                lengths.append(len(caption))
                words = caption.split()
                word_counts.append(len(words))
                all_words.extend(words)

                # Language detection
                lang = self.text_processor.detect_language(caption)
                languages.append(lang)

            except Exception as e:
                logger.warning(f"Failed to analyze caption {idx}: {e}")

        # Word frequency analysis (top 20)
        from collections import Counter

        word_freq = Counter(all_words).most_common(20)

        return {
            "sample_size": len(lengths),
            "length": {
                "min": min(lengths) if lengths else 0,
                "max": max(lengths) if lengths else 0,
                "avg": sum(lengths) / len(lengths) if lengths else 0,
            },
            "word_count": {
                "min": min(word_counts) if word_counts else 0,
                "max": max(word_counts) if word_counts else 0,
                "avg": sum(word_counts) / len(word_counts) if word_counts else 0,
            },
            "languages": {lang: languages.count(lang) for lang in set(languages)},
            "most_common_words": word_freq,
        }

    def validate_dataset(self, dataset_config: DatasetConfig) -> List[str]:
        """Validate dataset configuration and data"""
        issues = []

        # Check if dataset path exists
        dataset_path = Path(dataset_config.path)
        if not dataset_path.exists():
            issues.append(f"Dataset path does not exist: {dataset_path}")
            return issues

        try:
            # Try to load a small sample
            if dataset_config.type == "imagefolder":
                dataset = load_dataset(
                    "imagefolder", data_dir=dataset_config.path, split="train"
                )
            elif dataset_config.type == "json":
                with open(dataset_config.path, "r") as f:
                    data = json.load(f)
                dataset = HFDataset.from_list(data[:10])  # Sample first 10
            elif dataset_config.type == "parquet":
                df = pd.read_parquet(dataset_config.path)
                dataset = HFDataset.from_pandas(df.head(10))
            else:
                issues.append(f"Unsupported dataset type: {dataset_config.type}")
                return issues

            # Check required columns
            if dataset_config.caption_column not in dataset.column_names:
                issues.append(
                    f"Caption column '{dataset_config.caption_column}' not found"
                )

            if dataset_config.image_column not in dataset.column_names:
                issues.append(f"Image column '{dataset_config.image_column}' not found")

            # Validate sample data
            if len(dataset) > 0:
                try:
                    sample = dataset[0]

                    # Check image
                    image = sample[dataset_config.image_column]
                    if isinstance(image, str):
                        image_path = Path(image)
                        if not image_path.is_absolute():
                            image_path = dataset_path.parent / image
                        if not image_path.exists():
                            issues.append(f"Sample image file not found: {image_path}")
                        else:
                            # Try to load image
                            Image.open(image_path)

                    # Check caption
                    caption = sample[dataset_config.caption_column]
                    if not caption or (
                        isinstance(caption, str) and len(caption.strip()) == 0
                    ):
                        issues.append("Sample caption is empty")

                except Exception as e:
                    issues.append(f"Failed to validate sample data: {e}")

            else:
                issues.append("Dataset is empty")

        except Exception as e:
            issues.append(f"Failed to load dataset: {e}")

        return issues

    def prepare_dataset_for_training(
        self, config: TrainingConfig
    ) -> Tuple[TrainingDataset, Optional[TrainingDataset]]:
        """Prepare training and validation datasets"""
        from transformers import CLIPTokenizer

        # Load tokenizer
        tokenizer = CLIPTokenizer.from_pretrained(
            config.model.base_model_id,
            subfolder="tokenizer",
            revision=config.model.revision,
        )

        # Create training dataset
        train_dataset = TrainingDataset(
            config=config.dataset,
            tokenizer=tokenizer,
            resolution=config.resolution,
            augment=True,
        )

        # Create validation dataset if validation split is specified
        val_dataset = None
        if config.dataset.validation_split > 0:
            # Create validation dataset config
            val_config = DatasetConfig(**asdict(config.dataset))
            val_config.split = "validation"

            try:
                val_dataset = TrainingDataset(
                    config=val_config,
                    tokenizer=tokenizer,
                    resolution=config.resolution,
                    augment=False,  # No augmentation for validation
                )
            except Exception as e:
                logger.warning(f"⚠️ Failed to create validation dataset: {e}")

        return train_dataset, val_dataset


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
