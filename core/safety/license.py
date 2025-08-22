# core/safety/license.py
import os
import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class LicenseInfo:
    """License information structure"""

    license_type: str  # CC0, CC-BY, CC-BY-SA, MIT, Apache, Custom, etc.
    attribution_required: bool
    commercial_use: bool
    derivative_works: bool
    share_alike: bool
    source_url: Optional[str] = None
    author: Optional[str] = None
    license_text: Optional[str] = None
    expiry_date: Optional[str] = None
    restrictions: List[str] = None

    def __post_init__(self):
        if self.restrictions is None:
            self.restrictions = []


@dataclass
class UploadMetadata:
    """Complete upload metadata with license info"""

    file_id: str
    original_filename: str
    file_hash: str
    upload_timestamp: str
    uploader_id: str
    file_size: int
    content_type: str
    license_info: LicenseInfo
    safety_check: Dict[str, Any]
    processing_notes: List[str] = None

    def __post_init__(self):
        if self.processing_notes is None:
            self.processing_notes = []


class LicenseValidator:
    """Validate and categorize different license types"""

    KNOWN_LICENSES = {
        "CC0": {
            "full_name": "Creative Commons Zero",
            "attribution_required": False,
            "commercial_use": True,
            "derivative_works": True,
            "share_alike": False,
            "url": "https://creativecommons.org/publicdomain/zero/1.0/",
        },
        "CC-BY": {
            "full_name": "Creative Commons Attribution",
            "attribution_required": True,
            "commercial_use": True,
            "derivative_works": True,
            "share_alike": False,
            "url": "https://creativecommons.org/licenses/by/4.0/",
        },
        "CC-BY-SA": {
            "full_name": "Creative Commons Attribution-ShareAlike",
            "attribution_required": True,
            "commercial_use": True,
            "derivative_works": True,
            "share_alike": True,
            "url": "https://creativecommons.org/licenses/by-sa/4.0/",
        },
        "CC-BY-NC": {
            "full_name": "Creative Commons Attribution-NonCommercial",
            "attribution_required": True,
            "commercial_use": False,
            "derivative_works": True,
            "share_alike": False,
            "url": "https://creativecommons.org/licenses/by-nc/4.0/",
        },
        "MIT": {
            "full_name": "MIT License",
            "attribution_required": True,
            "commercial_use": True,
            "derivative_works": True,
            "share_alike": False,
            "url": "https://opensource.org/licenses/MIT",
        },
        "Apache-2.0": {
            "full_name": "Apache License 2.0",
            "attribution_required": True,
            "commercial_use": True,
            "derivative_works": True,
            "share_alike": False,
            "url": "https://opensource.org/licenses/Apache-2.0",
        },
        "Fair-Use": {
            "full_name": "Fair Use (Educational/Research)",
            "attribution_required": True,
            "commercial_use": False,
            "derivative_works": True,
            "share_alike": False,
            "restrictions": ["educational_only", "research_only"],
        },
        "Custom": {
            "full_name": "Custom License",
            "attribution_required": True,
            "commercial_use": False,
            "derivative_works": False,
            "share_alike": False,
            "restrictions": ["custom_terms_apply"],
        },
    }

    def validate_license(self, license_info: LicenseInfo) -> Dict[str, Any]:
        """Validate license information and return validation result"""
        result = {
            "is_valid": False,
            "license_recognized": False,
            "warnings": [],
            "requirements": [],
        }

        # Check if license type is recognized
        if license_info.license_type in self.KNOWN_LICENSES:
            result["license_recognized"] = True
            known_info = self.KNOWN_LICENSES[license_info.license_type]

            # Auto-fill missing information from known license
            if (
                not hasattr(license_info, "attribution_required")
                or license_info.attribution_required is None
            ):
                license_info.attribution_required = known_info["attribution_required"]
            if (
                not hasattr(license_info, "commercial_use")
                or license_info.commercial_use is None
            ):
                license_info.commercial_use = known_info["commercial_use"]
            if (
                not hasattr(license_info, "derivative_works")
                or license_info.derivative_works is None
            ):
                license_info.derivative_works = known_info["derivative_works"]
            if (
                not hasattr(license_info, "share_alike")
                or license_info.share_alike is None
            ):
                license_info.share_alike = known_info["share_alike"]

        # Validation checks
        if license_info.attribution_required and not license_info.author:
            result["warnings"].append("Attribution required but no author specified")

        if license_info.license_type == "Custom" and not license_info.license_text:
            result["warnings"].append("Custom license requires license text")

        # Check for restrictive combinations
        if not license_info.commercial_use:
            result["requirements"].append("Non-commercial use only")

        if license_info.share_alike:
            result["requirements"].append("Derivative works must use same license")

        if license_info.attribution_required:
            result["requirements"].append("Attribution required in all uses")

        # Consider valid if no critical errors
        result["is_valid"] = (
            len([w for w in result["warnings"] if "required" in w.lower()]) == 0
        )

        return result


class LicenseManager:
    """Manage license information and compliance tracking"""

    def __init__(self, cache_root: str):
        self.cache_root = Path(cache_root)
        self.license_db_path = self.cache_root / "metadata" / "licenses.jsonl"
        self.validator = LicenseValidator()

        # Ensure directories exist
        self.license_db_path.parent.mkdir(parents=True, exist_ok=True)

    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def register_upload(
        self,
        file_path: str,
        license_info: LicenseInfo,
        uploader_id: str,
        safety_check: Dict[str, Any],
    ) -> UploadMetadata:
        """Register a new file upload with license information"""

        file_path = Path(file_path)

        # Generate file metadata
        metadata = UploadMetadata(
            file_id=hashlib.md5(
                f"{file_path.name}{datetime.now().isoformat()}".encode()
            ).hexdigest(),
            original_filename=file_path.name,
            file_hash=self.calculate_file_hash(str(file_path)),
            upload_timestamp=datetime.now(timezone.utc).isoformat(),
            uploader_id=uploader_id,
            file_size=file_path.stat().st_size,
            content_type=self._guess_content_type(file_path),
            license_info=license_info,
            safety_check=safety_check,
        )

        # Validate license
        validation_result = self.validator.validate_license(license_info)
        if not validation_result["is_valid"]:
            raise ValueError(f"Invalid license: {validation_result['warnings']}")

        # Store in database
        self._store_metadata(metadata)

        logger.info(
            f"Registered upload: {metadata.file_id} ({metadata.original_filename})"
        )
        return metadata

    def check_usage_compliance(self, file_id: str, intended_use: str) -> Dict[str, Any]:
        """Check if intended use complies with file's license"""
        metadata = self.get_metadata(file_id)
        if not metadata:
            return {"compliant": False, "reason": "File not found"}

        license_info = metadata.license_info

        # Check different use cases
        compliance_checks = {
            "commercial": (
                license_info.commercial_use if intended_use == "commercial" else True
            ),
            "derivative": (
                license_info.derivative_works if "derivative" in intended_use else True
            ),
            "attribution": True,  # Always check attribution requirements
        }

        is_compliant = all(compliance_checks.values())

        requirements = []
        if license_info.attribution_required:
            requirements.append(f"Must attribute to: {license_info.author}")
        if license_info.share_alike and "derivative" in intended_use:
            requirements.append(
                f"Derivative works must use {license_info.license_type} license"
            )
        if not license_info.commercial_use and intended_use == "commercial":
            requirements.append("Commercial use not permitted")

        return {
            "compliant": is_compliant,
            "requirements": requirements,
            "license_type": license_info.license_type,
            "restrictions": license_info.restrictions,
        }

    def generate_attribution_text(self, file_id: str) -> str:
        """Generate proper attribution text for a file"""
        metadata = self.get_metadata(file_id)
        if not metadata or not metadata.license_info.attribution_required:
            return ""

        license_info = metadata.license_info
        attribution_parts = []

        if license_info.author:
            attribution_parts.append(f"Author: {license_info.author}")

        attribution_parts.append(f"License: {license_info.license_type}")

        if license_info.source_url:
            attribution_parts.append(f"Source: {license_info.source_url}")

        return " | ".join(attribution_parts)

    def get_metadata(self, file_id: str) -> Optional[UploadMetadata]:
        """Retrieve metadata for a file"""
        try:
            with open(self.license_db_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line.strip())
                    if data.get("file_id") == file_id:
                        # Reconstruct objects
                        license_info = LicenseInfo(**data["license_info"])
                        metadata = UploadMetadata(
                            **{**data, "license_info": license_info}
                        )
                        return metadata
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        return None

    def list_files_by_license(self, license_type: str) -> List[UploadMetadata]:
        """List all files with specific license type"""
        files = []
        try:
            with open(self.license_db_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line.strip())
                    if data.get("license_info", {}).get("license_type") == license_type:
                        license_info = LicenseInfo(**data["license_info"])
                        metadata = UploadMetadata(
                            **{**data, "license_info": license_info}
                        )
                        files.append(metadata)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        return files

    def _store_metadata(self, metadata: UploadMetadata):
        """Store metadata to JSONL database"""
        with open(self.license_db_path, "a", encoding="utf-8") as f:
            # Convert to dict for JSON serialization
            data = asdict(metadata)
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def _guess_content_type(self, file_path: Path) -> str:
        """Guess content type from file extension"""
        ext = file_path.suffix.lower()
        content_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".pdf": "application/pdf",
            ".zip": "application/zip",
        }
        return content_types.get(ext, "application/octet-stream")
