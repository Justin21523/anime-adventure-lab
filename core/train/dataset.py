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
from dataclasses import asdict
from transformers import CLIPTokenizer
import albumentations as A
from albumentations.pytorch import ToTensorV2

from ..config import get_config
from ..utils.image import ImageProcessor, get_image_processor
from ..utils.text import TextProcessor, get_text_processor
from ..exceptions import (
    DatasetError,
    ValidationError,
    DatasetLoadError,
    DatasetNotFoundError,
)
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
        self.length = len(self.dataset) if self.dataset else 0

        # Setup augmentations
        self.transforms = self._setup_augmentations() if augment else None

        logger.info(f"📚 Dataset loaded: {self.length} samples")

    def _load_dataset(self) -> Optional[Union[HFDataset, List[Dict[str, Any]]]]:
        """Load dataset from various sources"""
        if not self.config.path:
            raise DatasetError("Dataset path is required")

        try:
            if self.config.type == "imagefolder":
                return self._load_imagefolder()
            elif self.config.type == "json":
                return self._load_json()
            elif self.config.type == "parquet":
                return self._load_parquet()
            elif self.config.type == "huggingface":
                return self._load_huggingface()
            else:
                raise DatasetError(f"Unsupported dataset type: {self.config.type}")

        except Exception as e:
            raise DatasetLoadError(self.config.path, str(e))

    def _load_imagefolder(self) -> Union[HFDataset, List[Dict[str, Any]]]:
        """Load from image folder structure"""
        data_path = Path(self.config.path)
        if not data_path.exists():
            raise DatasetNotFoundError(str(data_path))

        # Use HuggingFace datasets if available
        dataset = load_dataset(
            "imagefolder", data_dir=str(data_path), split=self.config.split
        )
        return dataset  # type: ignore

    def _manual_load_imagefolder(self, data_path: Path) -> List[Dict[str, Any]]:
        """Manually load imagefolder when HF datasets not available"""
        data = []

        # Look for images in subdirectories
        for subdir in data_path.iterdir():
            if subdir.is_dir():
                caption_file = subdir / "caption.txt"

                # Fix: Handle different image extensions properly
                image_extensions = [
                    "*.jpg",
                    "*.jpeg",
                    "*.png",
                    "*.bmp",
                    "*.webp",
                    "*.JPG",
                    "*.JPEG",
                    "*.PNG",
                ]
                image_files = []
                for ext in image_extensions:
                    image_files.extend(subdir.glob(ext))

                for image_file in image_files:
                    item = {"image": str(image_file)}

                    # Try to load caption
                    if caption_file.exists():
                        try:
                            with open(caption_file, "r", encoding="utf-8") as f:
                                item["text"] = f.read().strip()
                        except Exception:
                            item["text"] = ""
                    else:
                        item["text"] = subdir.name  # Use folder name as caption

                    data.append(item)

        return data

    def _load_json(self) -> Union[HFDataset, List[Dict[str, Any]]]:
        """Load from JSON file"""
        json_path = Path(self.config.path)
        if not json_path.exists():
            raise DatasetNotFoundError(str(json_path))

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            return HFDataset.from_list(data)

        except Exception as e:
            raise DatasetLoadError(str(json_path), f"JSON parsing failed: {e}")

    def _load_parquet(self) -> Union[HFDataset, List[Dict[str, Any]]]:
        """Load from Parquet file"""
        parquet_path = Path(self.config.path)
        if not parquet_path.exists():
            raise DatasetNotFoundError(str(parquet_path))

        try:
            df = pd.read_parquet(parquet_path)
            data = df.to_dict("records")

            return HFDataset.from_pandas(df)

        except Exception as e:
            raise DatasetLoadError(str(parquet_path), f"Parquet loading failed: {e}")

    def _load_huggingface(self) -> HFDataset:
        """Load from Hugging Face Hub"""

        try:
            dataset = load_dataset(
                self.config.path,
                split=self.config.split,
                cache_dir=self.config.cache_dir,
            )
            return dataset  # type: ignore

        except Exception as e:
            raise DatasetLoadError(self.config.path, f"HF dataset loading failed: {e}")

    def _setup_augmentations(self) -> Optional[A.Compose]:
        """Setup image augmentations"""
        try:
            # Fix: Proper parameter types for albumentations
            transforms = A.Compose(
                [  # type: ignore
                    # Color augmentations (mild)
                    A.ColorJitter(
                        brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3
                    ),
                    # Geometric augmentations (very mild for character consistency)
                    A.HorizontalFlip(p=0.5),
                    # Noise and blur (very mild)
                    A.OneOf(
                        [
                            A.GaussNoise(var_limit=(0, 25), p=0.2),  # type: ignore
                            A.GaussianBlur(blur_limit=3, p=0.1),
                            A.MotionBlur(blur_limit=3, p=0.1),
                        ],
                        p=0.2,
                    ),
                    # Normalization for diffusion models
                    A.Normalize(
                        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0  # type: ignore
                    ),
                    ToTensorV2(),
                ]
            )
            return transforms

        except Exception as e:
            logger.warning(f"Failed to setup augmentations: {e}")
            return None

    def __len__(self) -> int:
        return self.length

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

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single item from the dataset"""
        try:
            # Get item from dataset
            if isinstance(self.dataset, list):
                item = self.dataset[idx]
            elif hasattr(self.dataset, "__getitem__"):
                item = self.dataset[idx]  # type: ignore
            else:
                raise DatasetError(f"Dataset type not supported: {type(self.dataset)}")

            # Process image
            image = self._process_image(item)

            # Process text
            text = self._process_text(item)

            result = {
                "image": image,
                "text": text,
                "original_size": (
                    (image.height, image.width)
                    if hasattr(image, "height")
                    else (512, 512)
                ),
            }

            # Add tokenized text if tokenizer is available
            if self.tokenizer and text:
                try:
                    tokens = self.tokenizer(
                        text,
                        truncation=True,
                        padding="max_length",
                        max_length=77,  # Standard CLIP length
                        return_tensors="pt",
                    )
                    result["input_ids"] = tokens.input_ids.squeeze()
                    result["attention_mask"] = tokens.attention_mask.squeeze()
                except Exception as e:
                    logger.warning(f"Tokenization failed for item {idx}: {e}")

            return result

        except Exception as e:
            logger.error(f"Failed to get item {idx}: {e}")
            # Return a fallback item
            return self._get_fallback_item()

    def _process_image(self, item: Dict[str, Any]) -> Image.Image:
        """Process image from dataset item"""
        image_key = self.config.image_column

        try:
            if image_key in item:
                image_data = item[image_key]

                # Handle different image formats
                if isinstance(image_data, str):
                    # File path
                    image_path = Path(image_data)
                    if not image_path.is_absolute():
                        # Make relative paths absolute
                        base_path = Path(self.config.path).parent
                        image_path = base_path / image_data

                    image = Image.open(image_path).convert("RGB")

                elif isinstance(image_data, Image.Image):
                    # PIL Image
                    image = image_data.convert("RGB")

                elif hasattr(image_data, "save"):
                    # HF Image object
                    image = image_data.convert("RGB")

                else:
                    raise DatasetError(f"Unsupported image format: {type(image_data)}")

                # Resize image
                if self.resolution != image.size[0] or self.resolution != image.size[1]:
                    image = image.resize(
                        (self.resolution, self.resolution), Image.Resampling.LANCZOS
                    )

                # Apply transforms if available
                if self.transforms:
                    # Convert to numpy for albumentations
                    image_np = np.array(image)
                    transformed = self.transforms(image=image_np)

                    # If transforms return tensor, convert back to PIL for consistency
                    if isinstance(transformed["image"], torch.Tensor):
                        # Convert from tensor back to PIL Image
                        tensor_img = transformed["image"]
                        if tensor_img.shape[0] == 3:  # CHW format
                            tensor_img = tensor_img.permute(1, 2, 0)  # HWC format

                        # Denormalize from [-1, 1] to [0, 255]
                        tensor_img = (tensor_img + 1.0) * 127.5
                        tensor_img = tensor_img.clamp(0, 255).byte()

                        image = Image.fromarray(tensor_img.numpy(), "RGB")
                    else:
                        image = Image.fromarray(transformed["image"], "RGB")

                return image

            else:
                raise DatasetError(f"Image column '{image_key}' not found in item")

        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            # Return a black placeholder image
            return Image.new("RGB", (self.resolution, self.resolution), color=(0, 0, 0))

    def _process_text(self, item: Dict[str, Any]) -> str:
        """Process text/caption from dataset item"""
        text_key = self.config.caption_column

        try:
            if text_key in item:
                text = item[text_key]
                if isinstance(text, str):
                    return text.strip()
                else:
                    return str(text).strip()
            else:
                logger.warning(f"Text column '{text_key}' not found in item")
                return ""

        except Exception as e:
            logger.error(f"Text processing failed: {e}")
            return ""

    def _get_fallback_item(self) -> Dict[str, Any]:
        """Return a fallback item when loading fails"""
        fallback_image = Image.new(
            "RGB", (self.resolution, self.resolution), color=(128, 128, 128)
        )

        result = {
            "image": fallback_image,
            "text": "fallback image",
            "original_size": (self.resolution, self.resolution),
        }

        if self.tokenizer:
            try:
                tokens = self.tokenizer(
                    "fallback image",
                    truncation=True,
                    padding="max_length",
                    max_length=77,
                    return_tensors="pt",
                )
                result["input_ids"] = tokens.input_ids.squeeze()
                result["attention_mask"] = tokens.attention_mask.squeeze()
            except Exception:
                pass

        return result


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
                "total_samples": len(dataset),  # type: ignore
                "columns": dataset.column_names,
                "features": dict(dataset.features),  # type: ignore
            }

            # Image analysis (sample-based for performance)
            if "image" in dataset.column_names:  # type: ignore
                image_stats = self._analyze_images(
                    dataset, sample_size=min(100, len(dataset))  # type: ignore
                )
                stats["images"] = image_stats

            # Text analysis
            if "text" in dataset.column_names:  # type: ignore
                text_stats = self._analyze_captions(
                    dataset, sample_size=min(1000, len(dataset))  # type: ignore
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
            if dataset_config.caption_column not in dataset.column_names:  # type: ignore
                issues.append(
                    f"Caption column '{dataset_config.caption_column}' not found"
                )

            if dataset_config.image_column not in dataset.column_names:  # type: ignore
                issues.append(f"Image column '{dataset_config.image_column}' not found")

            # Validate sample data
            if len(dataset) > 0:  # type: ignore
                try:
                    sample = dataset[0]  # type: ignore

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


class DatasetFactory:
    """Factory for creating training datasets"""

    @staticmethod
    def create_dataset(
        config: DatasetConfig,
        tokenizer: Optional[Any] = None,
        resolution: int = 512,
        augment: bool = True,
    ) -> TrainingDataset:
        """Create a training dataset from configuration"""
        return TrainingDataset(
            config=config,
            tokenizer=tokenizer,  # type: ignore
            resolution=resolution,
            augment=augment,
        )

    @staticmethod
    def create_dataloader(
        dataset: TrainingDataset,
        batch_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 0,
        **kwargs,
    ) -> DataLoader:
        """Create a DataLoader from dataset"""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,  # Important for training consistency
            **kwargs,
        )

    @staticmethod
    def validate_dataset_config(config: DatasetConfig) -> List[str]:
        """Validate dataset configuration"""
        warnings = []

        # Check path
        if not config.path:
            warnings.append("Dataset path is empty")
        elif not Path(config.path).exists():
            warnings.append(f"Dataset path does not exist: {config.path}")

        # Check type
        supported_types = ["imagefolder", "json", "parquet", "huggingface"]
        if config.type not in supported_types:
            warnings.append(f"Unsupported dataset type: {config.type}")

        # Check required dependencies
        if config.type == "huggingface":
            warnings.append(
                "HuggingFace datasets library required for huggingface type"
            )

        # Check validation split
        if not (0.0 <= config.validation_split <= 1.0):
            warnings.append("Validation split must be between 0.0 and 1.0")

        return warnings


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


# Utility functions


def get_dataset_info(dataset_path: str) -> Dict[str, Any]:
    """Get basic information about a dataset"""
    path = Path(dataset_path)

    info = {
        "path": str(path),
        "exists": path.exists(),
        "type": "unknown",
        "estimated_size": 0,
        "num_files": 0,
    }

    if not path.exists():
        return info

    try:
        if path.is_file():
            # Single file dataset
            if path.suffix.lower() == ".json":
                info["type"] = "json"
                with open(path, "r") as f:
                    data = json.load(f)
                    info["estimated_size"] = len(data) if isinstance(data, list) else 1
            elif path.suffix.lower() == ".parquet":
                info["type"] = "parquet"
                df = pd.read_parquet(path)
                info["estimated_size"] = len(df)

            info["num_files"] = 1

        elif path.is_dir():
            # Directory dataset
            info["type"] = "imagefolder"

            # Count image files
            image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            image_files = []

            for ext in image_extensions:
                image_files.extend(path.rglob(f"*{ext}"))
                image_files.extend(path.rglob(f"*{ext.upper()}"))

            info["num_files"] = len(image_files)
            info["estimated_size"] = len(image_files)

    except Exception as e:
        logger.error(f"Failed to analyze dataset {dataset_path}: {e}")

    return info
