"""Detected page regions shared between detection and later steps."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    """Pixel coordinates in the image artifact; right and bottom edges are exclusive."""

    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class TextRegion:
    box: BoundingBox
    confidence: float
    original_text: str | None = None
    translated_text: str | None = None
