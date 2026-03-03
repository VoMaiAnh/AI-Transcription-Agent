"""
File handling utilities
"""

import os
import time
from pathlib import Path
from typing import List, Optional


def safe_remove_file(
    file_path: str,
    max_retries: int = 3,
    delay: float = 0.5
) -> bool:
    """
    Safely remove a file with retries (for Windows file locking issues)

    Args:
        file_path: Path to file to remove
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds

    Returns:
        True if file was removed, False otherwise
    """
    path = Path(file_path)
    if not path.exists():
        return True

    for attempt in range(max_retries):
        try:
            os.remove(path)
            return True
        except (PermissionError, OSError):
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                print(f"Warning: Could not remove file {path} after {max_retries} attempts")
                return False

    return False


def get_file_size(file_path: str) -> int:
    """
    Get file size in bytes

    Args:
        file_path: Path to file

    Returns:
        File size in bytes
    """
    return os.path.getsize(file_path)


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def get_file_extension(filename: str) -> str:
    """
    Get file extension in lowercase

    Args:
        filename: Name of file

    Returns:
        Lowercase extension with dot (e.g., '.mp3')
    """
    return Path(filename).suffix.lower()


def generate_unique_filename(
    prefix: str,
    extension: str,
    include_timestamp: bool = True
) -> str:
    """
    Generate a unique filename

    Args:
        prefix: Filename prefix
        extension: File extension
        include_timestamp: Whether to include timestamp

    Returns:
        Unique filename
    """
    import uuid
    from datetime import datetime

    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}_{uuid.uuid4().hex[:8]}{extension}"
    else:
        return f"{prefix}_{uuid.uuid4().hex}{extension}"


def ensure_directory(directory: str) -> Path:
    """
    Ensure directory exists, create if necessary

    Args:
        directory: Directory path

    Returns:
        Path object for directory
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_files(
    directory: str,
    extensions: Optional[List[str]] = None,
    recursive: bool = False
) -> List[Path]:
    """
    List files in directory

    Args:
        directory: Directory to search
        extensions: Optional list of extensions to filter (e.g., ['.mp3', '.wav'])
        recursive: Whether to search recursively

    Returns:
        List of Path objects
    """
    path = Path(directory)

    if recursive:
        files = list(path.rglob("*"))
    else:
        files = list(path.glob("*"))

    # Filter to files only
    files = [f for f in files if f.is_file()]

    # Filter by extension if specified
    if extensions:
        extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}'
                      for ext in extensions]
        files = [f for f in files if f.suffix.lower() in extensions]

    return files
