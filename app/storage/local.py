"""
Local file storage backend
"""

import os
import shutil
from pathlib import Path
from typing import Optional, BinaryIO

from app.config import settings


class LocalStorage:
    """Local file storage backend"""

    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize local storage

        Args:
            base_dir: Base directory for storage. Uses settings.upload_dir if None.
        """
        self.base_dir = base_dir or settings.upload_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_file(
        self,
        content: bytes,
        filename: str,
        subdirectory: Optional[str] = None
    ) -> Path:
        """
        Save file to storage

        Args:
            content: File content as bytes
            filename: Name of the file
            subdirectory: Optional subdirectory

        Returns:
            Path to saved file
        """
        if subdirectory:
            dir_path = self.base_dir / subdirectory
            dir_path.mkdir(parents=True, exist_ok=True)
        else:
            dir_path = self.base_dir

        file_path = dir_path / filename

        with open(file_path, "wb") as f:
            f.write(content)

        return file_path

    def save_uploaded_file(
        self,
        file,
        filename: str,
        subdirectory: Optional[str] = None
    ) -> Path:
        """
        Save uploaded file to storage

        Args:
            file: UploadFile object
            filename: Name to save the file as
            subdirectory: Optional subdirectory

        Returns:
            Path to saved file
        """
        if subdirectory:
            dir_path = self.base_dir / subdirectory
            dir_path.mkdir(parents=True, exist_ok=True)
        else:
            dir_path = self.base_dir

        file_path = dir_path / filename

        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        return file_path

    def read_file(self, filename: str, subdirectory: Optional[str] = None) -> bytes:
        """
        Read file from storage

        Args:
            filename: Name of the file
            subdirectory: Optional subdirectory

        Returns:
            File content as bytes

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        file_path = self.get_file_path(filename, subdirectory)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "rb") as f:
            return f.read()

    def get_file_path(
        self,
        filename: str,
        subdirectory: Optional[str] = None
    ) -> Path:
        """
        Get path to file

        Args:
            filename: Name of the file
            subdirectory: Optional subdirectory

        Returns:
            Path to file
        """
        if subdirectory:
            return self.base_dir / subdirectory / filename
        return self.base_dir / filename

    def file_exists(
        self,
        filename: str,
        subdirectory: Optional[str] = None
    ) -> bool:
        """
        Check if file exists in storage

        Args:
            filename: Name of the file
            subdirectory: Optional subdirectory

        Returns:
            True if file exists, False otherwise
        """
        file_path = self.get_file_path(filename, subdirectory)
        return file_path.exists()

    def delete_file(
        self,
        filename: str,
        subdirectory: Optional[str] = None
    ) -> bool:
        """
        Delete file from storage

        Args:
            filename: Name of the file
            subdirectory: Optional subdirectory

        Returns:
            True if file was deleted, False otherwise
        """
        file_path = self.get_file_path(filename, subdirectory)

        if file_path.exists():
            file_path.unlink()
            return True
        return False

    def get_file_size(
        self,
        filename: str,
        subdirectory: Optional[str] = None
    ) -> int:
        """
        Get file size in bytes

        Args:
            filename: Name of the file
            subdirectory: Optional subdirectory

        Returns:
            File size in bytes

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        file_path = self.get_file_path(filename, subdirectory)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        return file_path.stat().st_size


# Global storage instance
storage = LocalStorage()


def get_storage() -> LocalStorage:
    """Get storage instance"""
    return storage
