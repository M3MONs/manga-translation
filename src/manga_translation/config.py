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
class DetectionOptions:
    model_path: Path = Path("models/comictextdetector.pt.onnx")
    """comic-text-detector ONNX weights, downloaded manually."""
    confidence_threshold: float = 0.4
    """Detections scoring below this are discarded."""
    nms_threshold: float = 0.35
    """IoU above which overlapping detections are merged into the best one."""

    def __post_init__(self) -> None:
        for name in ("confidence_threshold", "nms_threshold"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True)
class AppConfig:
    input_path: Path
    image: ImageOptions = field(default_factory=ImageOptions)
    detection: DetectionOptions = field(default_factory=DetectionOptions)
