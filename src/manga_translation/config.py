"""Application parameters grouped into immutable parameter objects."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ImageOptions:
    max_size: int = 4096
    """Longest allowed image side in pixels; 0 disables resizing."""

    def __post_init__(self) -> None:
        if self.max_size < 0:
            raise ValueError("max_size must be >= 0")


@dataclass(frozen=True)
class AppConfig:
    input_path: Path
    image: ImageOptions = field(default_factory=ImageOptions)
