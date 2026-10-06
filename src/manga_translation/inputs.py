"""Collection of the image files to translate."""

from pathlib import Path

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


class InputError(ValueError):
    """Raised when the input path does not provide any supported image."""


def _is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def _single_file(path: Path) -> list[Path]:
    """Return the file itself if it is supported, otherwise raise an error."""
    if not _is_supported(path):
        raise InputError(f"unsupported image format: {path}")
    return [path]


def _directory(path: Path) -> list[Path]:
    """Return the supported files directly inside a directory, sorted by name."""
    paths = sorted(child for child in path.iterdir() if child.is_file() and _is_supported(child))
    if not paths:
        raise InputError(f"no supported images in directory: {path}")
    return paths


def collect_image_paths(path: Path) -> list[Path]:
    """Return the file itself or the supported files directly inside a directory."""
    if path.is_file():
        return _single_file(path)
    return _directory(path)
