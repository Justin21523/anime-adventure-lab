# core/utils/image.py
"""
Image Processing Utilities
Standardized image handling across all modules
"""

import base64
import io
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Union, Tuple, Optional
from PIL import Image, ImageOps
import numpy as np

from ..exceptions import ImageProcessingError, ValidationError

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Standardized image processing utilities"""

    def __init__(self, max_size: int = 1024, quality: int = 95):
        self.max_size = max_size
        self.quality = quality

    def load_image(self, image: Union[str, bytes, Image.Image, Path]) -> Image.Image:
        """Load image from various formats with standardized preprocessing"""
        try:
            if isinstance(image, Image.Image):
                pil_image = image
            elif isinstance(image, (str, Path)):
                image_path = Path(image)
                if str(image).startswith("data:image"):
                    # Base64 data URL
                    header, data = str(image).split(",", 1)
                    image_data = base64.b64decode(data)
                    pil_image = Image.open(io.BytesIO(image_data))
                elif image_path.exists():
                    # File path
                    pil_image = Image.open(image_path)
                else:
                    raise ImageProcessingError(f"Image file not found: {image}")
            elif isinstance(image, bytes):
                pil_image = Image.open(io.BytesIO(image))
            else:
                raise ImageProcessingError(f"Unsupported image format: {type(image)}")

            # Convert to RGB and validate
            pil_image = pil_image.convert("RGB")

            # Validate image size
            width, height = pil_image.size
            if width < 32 or height < 32:
                raise ValidationError(
                    "image_size", f"{width}x{height}", "Image too small (minimum 32x32)"
                )

            # Resize if too large
            if max(width, height) > self.max_size:
                pil_image = self.resize_image(pil_image, self.max_size)
                logger.info(f"Image resized from {width}x{height} to {pil_image.size}")

            return pil_image

        except Exception as e:
            logger.error(f"Image loading failed: {e}")
            raise ImageProcessingError(str(e))

    def resize_image(
        self, image: Image.Image, max_size: int, maintain_aspect: bool = True
    ) -> Image.Image:
        """Resize image while maintaining aspect ratio"""
        if maintain_aspect:
            # Calculate new size maintaining aspect ratio
            width, height = image.size
            if width > height:
                new_width = max_size
                new_height = int(height * max_size / width)
            else:
                new_height = max_size
                new_width = int(width * max_size / height)

            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        else:
            return image.resize((max_size, max_size), Image.Resampling.LANCZOS)

    def crop_center(self, image: Image.Image, size: Tuple[int, int]) -> Image.Image:
        """Center crop image to specified size"""
        return ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    def to_base64(self, image: Image.Image, format: str = "JPEG") -> str:
        """Convert PIL Image to base64 string"""
        buffer = io.BytesIO()
        image.save(buffer, format=format, quality=self.quality)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/{format.lower()};base64,{encoded}"

    def from_base64(self, base64_str: str) -> Image.Image:
        """Convert base64 string to PIL Image"""
        if base64_str.startswith("data:image"):
            header, data = base64_str.split(",", 1)
            image_data = base64.b64decode(data)
        else:
            image_data = base64.b64decode(base64_str)

        return Image.open(io.BytesIO(image_data)).convert("RGB")

    def calculate_hash(self, image: Image.Image) -> str:
        """Calculate perceptual hash for duplicate detection"""
        # Simple implementation - can be enhanced with imagehash library
        resized = image.resize((8, 8), Image.Resampling.LANCZOS).convert("L")
        pixels = list(resized.getdata())
        avg = sum(pixels) / len(pixels)
        hash_bits = "".join("1" if pixel > avg else "0" for pixel in pixels)
        return hashlib.md5(hash_bits.encode()).hexdigest()[:16]

    def apply_safety_blur(
        self, image: Image.Image, blur_regions: List[Tuple[int, int, int, int]]
    ) -> Image.Image:
        """Apply blur to sensitive regions (face detection, etc.)"""
        from PIL import ImageFilter

        result_image = image.copy()

        for x1, y1, x2, y2 in blur_regions:
            # Extract region
            region = result_image.crop((x1, y1, x2, y2))

            # Apply strong blur
            blurred_region = region.filter(ImageFilter.GaussianBlur(radius=20))

            # Paste back
            result_image.paste(blurred_region, (x1, y1))

        return result_image

    def validate_image_content(self, image: Image.Image) -> Dict[str, Any]:
        """Basic image content validation"""
        width, height = image.size

        # Check if image is mostly black/white
        grayscale = image.convert("L")
        histogram = grayscale.histogram()

        # Count dark/light pixels
        dark_pixels = sum(histogram[:85])  # 0-84 (very dark)
        light_pixels = sum(histogram[170:])  # 170-255 (very light)
        total_pixels = width * height

        validation_result = {
            "is_valid": True,
            "width": width,
            "height": height,
            "aspect_ratio": round(width / height, 2),
            "dark_ratio": round(dark_pixels / total_pixels, 3),
            "light_ratio": round(light_pixels / total_pixels, 3),
            "warnings": [],
        }

        # Add warnings for potential issues
        if validation_result["dark_ratio"] > 0.8:
            validation_result["warnings"].append("Image appears very dark")
        elif validation_result["light_ratio"] > 0.8:
            validation_result["warnings"].append("Image appears very bright/washed out")

        if width < 224 or height < 224:
            validation_result["warnings"].append(
                "Image resolution may be too low for optimal results"
            )

        return validation_result

    def create_thumbnail(
        self, image: Image.Image, size: Tuple[int, int] = (256, 256)
    ) -> Image.Image:
        """Create thumbnail for previews"""
        thumbnail = image.copy()
        thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
        return thumbnail

    def get_image_stats(self, image: Image.Image) -> Dict[str, Any]:
        """Get comprehensive image statistics"""
        width, height = image.size

        # Convert to numpy for analysis
        img_array = np.array(image)

        stats = {
            "format": image.format or "Unknown",
            "mode": image.mode,
            "size": {"width": width, "height": height},
            "aspect_ratio": round(width / height, 3),
            "channels": len(img_array.shape) if len(img_array.shape) > 2 else 1,
            "pixel_count": width * height,
            "file_size_estimate_kb": round(
                width * height * 3 / 1024, 1
            ),  # RGB estimate
        }

        # Color statistics
        if image.mode == "RGB":
            r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
            stats["color_stats"] = {
                "mean_rgb": [float(r.mean()), float(g.mean()), float(b.mean())],
                "std_rgb": [float(r.std()), float(g.std()), float(b.std())],
                "brightness": float(img_array.mean()),
                "contrast": float(img_array.std()),
            }

        return stats


# Global image processor
_image_processor: Optional[ImageProcessor] = None


def get_image_processor() -> ImageProcessor:
    """Get global image processor instance"""
    global _image_processor
    if _image_processor is None:
        _image_processor = ImageProcessor()
    return _image_processor
