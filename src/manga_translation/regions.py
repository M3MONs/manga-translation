"""Detected page regions shared between detection and later steps."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    """Pixel coordinates in the image artifact; right and bottom edges are exclusive."""

    x1: int
    y1: int
    x2: int
    y2: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        """(x1, y1, x2, y2), the order PIL uses for crop boxes."""
        return self.x1, self.y1, self.x2, self.y2


@dataclass(frozen=True)
class TextRegion:
    box: BoundingBox
    confidence: float
    original_text: str | None = None
    translated_text: str | None = None
