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
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import numpy as np
import cv2

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
                raise ImageProcessingError(f"Unsupported image type: {type(image)}")

            # Convert to RGB if needed
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")

            return pil_image

        except Exception as e:
            raise ImageProcessingError(f"Failed to load image: {e}")

    def resize_image(
        self,
        image: Image.Image,
        max_size: Optional[int] = None,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Image.Image:
        """Resize image while maintaining aspect ratio or to exact size"""
        if target_size:
            return image.resize(target_size, Image.Resampling.LANCZOS)

        max_size = max_size or self.max_size
        width, height = image.size

        if width <= max_size and height <= max_size:
            return image

        # Calculate new size maintaining aspect ratio
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))

        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def normalize_image(
        self,
        image: Image.Image,
        mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> np.ndarray:
        """Normalize image for model input"""
        # Convert to numpy array
        img_array = np.array(image).astype(np.float32) / 255.0

        # Normalize
        img_array = (img_array - np.array(mean)) / np.array(std)

        # Convert to CHW format
        img_array = np.transpose(img_array, (2, 0, 1))

        return img_array

    def denormalize_image(
        self,
        tensor: np.ndarray,
        mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> Image.Image:
        """Denormalize tensor back to PIL Image"""
        # Convert from CHW to HWC
        if tensor.ndim == 3 and tensor.shape[0] == 3:
            tensor = np.transpose(tensor, (1, 2, 0))

        # Denormalize
        tensor = tensor * np.array(std) + np.array(mean)

        # Clamp and convert to uint8
        tensor = np.clip(tensor * 255, 0, 255).astype(np.uint8)

        return Image.fromarray(tensor)

    def center_crop(self, image: Image.Image, size: Tuple[int, int]) -> Image.Image:
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

    def calculate_hash(self, image: Image.Image, hash_size: int = 8) -> str:
        """Calculate perceptual hash for duplicate detection"""
        # Resize to hash_size x hash_size and convert to grayscale
        resized = image.resize(
            (hash_size, hash_size), Image.Resampling.LANCZOS
        ).convert("L")
        pixels = list(resized.getdata())

        # Calculate average
        avg = sum(pixels) / len(pixels)

        # Create hash bits
        hash_bits = "".join("1" if pixel > avg else "0" for pixel in pixels)

        # Convert to hex
        return hashlib.md5(hash_bits.encode()).hexdigest()[:16]

    def detect_faces(self, image: Image.Image) -> List[Tuple[int, int, int, int]]:
        """Detect faces in image using OpenCV"""
        try:
            # Convert PIL to OpenCV format
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

            # Load face cascade
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            # Convert to (x1, y1, x2, y2) format
            face_boxes = []
            for x, y, w, h in faces:
                face_boxes.append((x, y, x + w, y + h))

            return face_boxes

        except Exception as e:
            logger.warning(f"Face detection failed: {e}")
            return []

    def apply_safety_blur(
        self,
        image: Image.Image,
        blur_regions: List[Tuple[int, int, int, int]],
        blur_radius: int = 20,
    ) -> Image.Image:
        """Apply blur to sensitive regions"""
        result_image = image.copy()

        for x1, y1, x2, y2 in blur_regions:
            # Extract region
            region = result_image.crop((x1, y1, x2, y2))

            # Apply blur
            blurred_region = region.filter(ImageFilter.GaussianBlur(radius=blur_radius))

            # Paste back
            result_image.paste(blurred_region, (x1, y1))

        return result_image

    def enhance_image(
        self,
        image: Image.Image,
        brightness: float = 1.0,
        contrast: float = 1.0,
        saturation: float = 1.0,
        sharpness: float = 1.0,
    ) -> Image.Image:
        """Apply image enhancements"""
        enhanced = image.copy()

        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(brightness)

        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(contrast)

        if saturation != 1.0:
            enhancer = ImageEnhance.Color(enhanced)
            enhanced = enhancer.enhance(saturation)

        if sharpness != 1.0:
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(sharpness)

        return enhanced

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

    def apply_watermark(
        self,
        image: Image.Image,
        watermark_text: str = "Generated by AI",
        position: str = "bottom-right",
        opacity: float = 0.5,
    ) -> Image.Image:
        """Apply text watermark to image"""
        try:
            from PIL import ImageDraw, ImageFont

            # Create copy
            watermarked = image.copy()

            # Create transparent overlay
            overlay = Image.new("RGBA", watermarked.size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(overlay)

            # Try to load a font
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                font = ImageFont.load_default()

            # Get text size
            bbox = draw.textbbox((0, 0), watermark_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Calculate position
            if position == "bottom-right":
                x = watermarked.width - text_width - 20
                y = watermarked.height - text_height - 20
            elif position == "bottom-left":
                x = 20
                y = watermarked.height - text_height - 20
            elif position == "top-right":
                x = watermarked.width - text_width - 20
                y = 20
            elif position == "top-left":
                x = 20
                y = 20
            else:  # center
                x = (watermarked.width - text_width) // 2
                y = (watermarked.height - text_height) // 2

            # Draw text
            alpha = int(255 * opacity)
            draw.text((x, y), watermark_text, font=font, fill=(255, 255, 255, alpha))

            # Composite
            watermarked = Image.alpha_composite(watermarked.convert("RGBA"), overlay)
            return watermarked.convert("RGB")

        except Exception as e:
            logger.warning(f"Failed to apply watermark: {e}")
            return image

    def auto_adjust(self, image: Image.Image) -> Image.Image:
        """Automatically adjust image brightness, contrast and color"""
        try:
            # Convert to numpy for analysis
            img_array = np.array(image)

            # Calculate automatic adjustments
            brightness_factor = 1.0
            contrast_factor = 1.0

            # Brightness adjustment based on mean
            mean_brightness = img_array.mean()
            if mean_brightness < 100:  # Too dark
                brightness_factor = 1.2
            elif mean_brightness > 180:  # Too bright
                brightness_factor = 0.9

            # Contrast adjustment based on standard deviation
            contrast_std = img_array.std()
            if contrast_std < 40:  # Low contrast
                contrast_factor = 1.3
            elif contrast_std > 80:  # High contrast
                contrast_factor = 0.9

            # Apply adjustments
            enhanced = self.enhance_image(
                image, brightness=brightness_factor, contrast=contrast_factor
            )

            return enhanced

        except Exception as e:
            logger.warning(f"Auto adjustment failed: {e}")
            return image


# Global image processor
_image_processor: Optional[ImageProcessor] = None


def get_image_processor() -> ImageProcessor:
    """Get global image processor instance"""
    global _image_processor
    if _image_processor is None:
        _image_processor = ImageProcessor()
    return _image_processor
