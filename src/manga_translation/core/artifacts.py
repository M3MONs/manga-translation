"""Typed accessors for the artifacts steps exchange, since artifacts are stored as object.

Steps import these instead of each other, so each step only depends on the artifacts it reads.
"""

from pathlib import Path
from typing import cast

from PIL import Image

from .context import PageContext
from .regions import TextRegion


def get_image(context: PageContext) -> Image.Image:
    """The page as a uniform RGB image."""
    return cast(Image.Image, context.artifacts["image"])


def get_text_regions(context: PageContext) -> list[TextRegion]:
    return cast(list[TextRegion], context.artifacts["text_regions"])


def get_text_mask(context: PageContext) -> Image.Image:
    """Per-pixel text probability scaled to 0-255, mode L, same size as the page."""
    return cast(Image.Image, context.artifacts["text_mask"])


def get_output_image(context: PageContext) -> Image.Image:
    """Copy of the page that later steps edit; the image artifact stays unchanged."""
    return cast(Image.Image, context.artifacts["output_image"])


def get_output_path(context: PageContext) -> Path:
    """File the translated page was saved to."""
    return cast(Path, context.artifacts["output_path"])
