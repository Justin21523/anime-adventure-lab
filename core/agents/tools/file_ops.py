# core/agent/tools/file_ops.py
"""
File Operations Tool
Provides safe file system operations for agents
"""

import logging
import os
import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import mimetypes
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)


class SafeFileOperations:
    """
    Safe file operations with security restrictions
    Prevents access outside allowed directories
    """

    def __init__(self, allowed_dirs: Optional[List[str]] = None):
        # Default allowed directories (relative to current working directory)
        self.allowed_dirs = allowed_dirs or [
            ".",
            "./data",
            "./temp",
            "./outputs",
            "./cache",
        ]

        # Convert to absolute paths
        self.allowed_paths = []
        for dir_path in self.allowed_dirs:
            abs_path = os.path.abspath(dir_path)
            self.allowed_paths.append(abs_path)

        # File size limits (in bytes)
        self.max_file_size = 100 * 1024 * 1024  # 100MB
        self.max_files_per_operation = 1000

        # Restricted file extensions for security
        self.restricted_extensions = {
            ".exe",
            ".bat",
            ".cmd",
            ".com",
            ".pif",
            ".scr",
            ".vbs",
            ".js",
            ".jar",
            ".app",
            ".deb",
            ".rpm",
            ".dmg",
            ".pkg",
            ".msi",
        }

    def _is_path_allowed(self, file_path: str) -> bool:
        """Check if file path is within allowed directories"""
        try:
            abs_path = os.path.abspath(file_path)

            for allowed_path in self.allowed_paths:
                if abs_path.startswith(allowed_path):
                    return True

            logger.warning(
                f"Access denied to path outside allowed directories: {abs_path}"
            )
            return False

        except Exception as e:
            logger.error(f"Path validation error: {e}")
            return False

    def _is_extension_allowed(self, file_path: str) -> bool:
        """Check if file extension is allowed"""
        ext = os.path.splitext(file_path)[1].lower()
        return ext not in self.restricted_extensions

    def _get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get file information safely"""
        try:
            stat = os.stat(file_path)
            mime_type, _ = mimetypes.guess_type(file_path)

            return {
                "path": file_path,
                "name": os.path.basename(file_path),
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "is_file": os.path.isfile(file_path),
                "is_dir": os.path.isdir(file_path),
                "extension": os.path.splitext(file_path)[1],
                "mime_type": mime_type,
                "readable": os.access(file_path, os.R_OK),
                "writable": os.access(file_path, os.W_OK),
            }

        except Exception as e:
            logger.error(f"Error getting file info for {file_path}: {e}")
            return {"error": str(e)}


def list_files(path: str = ".", pattern: str = "*") -> Dict[str, Any]:
    """
    List files in directory

    Args:
        path: Directory path to list
        pattern: File pattern filter (e.g., "*.txt", "*.py")

    Returns:
        Dictionary with file listing results
    """
    try:
        file_ops = SafeFileOperations()

        if not file_ops._is_path_allowed(path):
            return {
                "success": False,
                "error": "Access denied: Path not in allowed directories",
            }

        if not os.path.exists(path):
            return {"success": False, "error": f"Path does not exist: {path}"}

        if not os.path.isdir(path):
            return {"success": False, "error": f"Path is not a directory: {path}"}

        # List files with pattern matching
        path_obj = Path(path)

        try:
            if pattern == "*":
                files = list(path_obj.iterdir())
            else:
                files = list(path_obj.glob(pattern))

            files = files[: file_ops.max_files_per_operation]  # Limit results

        except Exception as e:
            return {"success": False, "error": f"Error listing files: {str(e)}"}

        # Format file information
        file_list = []
        dir_list = []

        for file_path in files:
            file_info = file_ops._get_file_info(str(file_path))

            if "error" in file_info:
                continue

            if file_info["is_dir"]:
                dir_list.append(file_info)
            else:
                file_list.append(file_info)

        # Sort by name
        file_list.sort(key=lambda x: x["name"])
        dir_list.sort(key=lambda x: x["name"])

        return {
            "success": True,
            "path": path,
            "pattern": pattern,
            "directories": dir_list,
            "files": file_list,
            "total_dirs": len(dir_list),
            "total_files": len(file_list),
            "summary": f"Found {len(dir_list)} directories and {len(file_list)} files in {path}",
        }

    except Exception as e:
        logger.error(f"File listing failed: {e}")
        return {"success": False, "error": f"File listing failed: {str(e)}"}


def read_file(
    file_path: str, encoding: str = "utf-8", max_lines: int = 1000
) -> Dict[str, Any]:
    """
    Read file content safely

    Args:
        file_path: Path to file to read
        encoding: Text encoding (default: utf-8)
        max_lines: Maximum lines to read (for large files)

    Returns:
        Dictionary with file content or error
    """
    try:
        file_ops = SafeFileOperations()

        if not file_ops._is_path_allowed(file_path):
            return {
                "success": False,
                "error": "Access denied: Path not in allowed directories",
            }

        if not os.path.exists(file_path):
            return {"success": False, "error": f"File does not exist: {file_path}"}

        if not os.path.isfile(file_path):
            return {"success": False, "error": f"Path is not a file: {file_path}"}

        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size > file_ops.max_file_size:
            return {
                "success": False,
                "error": f"File too large: {file_size} bytes (max: {file_ops.max_file_size})",
            }

        # Get file info
        file_info = file_ops._get_file_info(file_path)

        # Try to read as text
        try:
            with open(file_path, "r", encoding=encoding) as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    lines.append(line.rstrip())

                content = "\n".join(lines)

                return {
                    "success": True,
                    "file_path": file_path,
                    "content": content,
                    "lines_read": len(lines),
                    "encoding": encoding,
                    "file_info": file_info,
                    "truncated": i >= max_lines,
                }

        except UnicodeDecodeError:
            # Try binary read for non-text files
            with open(file_path, "rb") as f:
                data = f.read(1024)  # Read first 1KB

                return {
                    "success": True,
                    "file_path": file_path,
                    "content": f"<Binary file - {len(data)} bytes shown>",
                    "binary_preview": data.hex()[:200],  # First 100 bytes as hex
                    "file_info": file_info,
                    "is_binary": True,
                }

    except Exception as e:
        logger.error(f"File read failed: {e}")
        return {"success": False, "error": f"File read failed: {str(e)}"}


def write_file(
    file_path: str, content: str, encoding: str = "utf-8", overwrite: bool = False
) -> Dict[str, Any]:
    """
    Write content to file safely

    Args:
        file_path: Path to file to write
        content: Content to write
        encoding: Text encoding (default: utf-8)
        overwrite: Whether to overwrite existing files

    Returns:
        Dictionary with write operation result
    """
    try:
        file_ops = SafeFileOperations()

        if not file_ops._is_path_allowed(file_path):
            return {
                "success": False,
                "error": "Access denied: Path not in allowed directories",
            }

        if not file_ops._is_extension_allowed(file_path):
            return {
                "success": False,
                "error": f"File extension not allowed: {os.path.splitext(file_path)[1]}",
            }

        # Check if file exists
        if os.path.exists(file_path) and not overwrite:
            return {
                "success": False,
                "error": f"File already exists: {file_path} (use overwrite=True to replace)",
            }

        # Check content size
        content_size = len(content.encode(encoding))
        if content_size > file_ops.max_file_size:
            return {
                "success": False,
                "error": f"Content too large: {content_size} bytes (max: {file_ops.max_file_size})",
            }

        # Create directory if needed
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Write file
        with open(file_path, "w", encoding=encoding) as f:
            f.write(content)

        # Get file info
        file_info = file_ops._get_file_info(file_path)

        return {
            "success": True,
            "file_path": file_path,
            "bytes_written": content_size,
            "lines_written": content.count("\n") + 1,
            "encoding": encoding,
            "file_info": file_info,
            "message": f"Successfully wrote {content_size} bytes to {file_path}",
        }

    except Exception as e:
        logger.error(f"File write failed: {e}")
        return {"success": False, "error": f"File write failed: {str(e)}"}


def file_exists(file_path: str) -> Dict[str, Any]:
    """
    Check if file exists and get basic info

    Args:
        file_path: Path to check

    Returns:
        Dictionary with existence check result
    """
    try:
        file_ops = SafeFileOperations()

        if not file_ops._is_path_allowed(file_path):
            return {
                "success": False,
                "error": "Access denied: Path not in allowed directories",
            }

        exists = os.path.exists(file_path)

        result = {"success": True, "file_path": file_path, "exists": exists}

        if exists:
            file_info = file_ops._get_file_info(file_path)
            result["file_info"] = file_info

        return result

    except Exception as e:
        logger.error(f"File existence check failed: {e}")
        return {"success": False, "error": f"File existence check failed: {str(e)}"}


def delete_file(file_path: str, confirm: bool = False) -> Dict[str, Any]:
    """
    Delete file safely

    Args:
        file_path: Path to file to delete
        confirm: Confirmation flag (required for safety)

    Returns:
        Dictionary with deletion result
    """
    try:
        if not confirm:
            return {
                "success": False,
                "error": "Deletion requires explicit confirmation (confirm=True)",
            }

        file_ops = SafeFileOperations()

        if not file_ops._is_path_allowed(file_path):
            return {
                "success": False,
                "error": "Access denied: Path not in allowed directories",
            }

        if not os.path.exists(file_path):
            return {"success": False, "error": f"File does not exist: {file_path}"}

        # Get file info before deletion
        file_info = file_ops._get_file_info(file_path)

        # Delete file
        os.remove(file_path)

        return {
            "success": True,
            "file_path": file_path,
            "deleted_file_info": file_info,
            "message": f"Successfully deleted {file_path}",
        }

    except Exception as e:
        logger.error(f"File deletion failed: {e}")
        return {"success": False, "error": f"File deletion failed: {str(e)}"}


def create_directory(dir_path: str, parents: bool = True) -> Dict[str, Any]:
    """
    Create directory safely

    Args:
        dir_path: Directory path to create
        parents: Create parent directories if needed

    Returns:
        Dictionary with creation result
    """
    try:
        file_ops = SafeFileOperations()

        if not file_ops._is_path_allowed(dir_path):
            return {
                "success": False,
                "error": "Access denied: Path not in allowed directories",
            }

        if os.path.exists(dir_path):
            if os.path.isdir(dir_path):
                return {
                    "success": True,
                    "dir_path": dir_path,
                    "message": f"Directory already exists: {dir_path}",
                }
            else:
                return {
                    "success": False,
                    "error": f"Path exists but is not a directory: {dir_path}",
                }

        # Create directory
        os.makedirs(dir_path, exist_ok=True) if parents else os.mkdir(dir_path)

        # Get directory info
        dir_info = file_ops._get_file_info(dir_path)

        return {
            "success": True,
            "dir_path": dir_path,
            "dir_info": dir_info,
            "message": f"Successfully created directory: {dir_path}",
        }

    except Exception as e:
        logger.error(f"Directory creation failed: {e}")
        return {"success": False, "error": f"Directory creation failed: {str(e)}"}
