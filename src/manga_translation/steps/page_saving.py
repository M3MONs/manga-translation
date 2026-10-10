"""Steps that write the translated page to a new file, leaving the source untouched."""

from pathlib import Path
from typing import cast

from PIL import Image

from ..config import OutputOptions
from ..context import PageContext
from ..step import Step, StepGroup
from .image_loading import get_image


def get_output_path(context: PageContext) -> Path:
    """Return the output_path artifact with its type, since artifacts are stored as object."""
    return cast(Path, context.artifacts["output_path"])


class SavePage:
    """Requires output_image (or image until text is erased) and source_path; produces output_path."""

    def __init__(self, options: OutputOptions) -> None:
        self.options = options

    async def execute(self, context: PageContext) -> None:
        path = self.options.output_dir / context.source_path.name
        if path.resolve() == context.source_path.resolve():
            raise ValueError(f"Output path {path} would overwrite the source image")
        image = cast(Image.Image, context.artifacts.get("output_image", get_image(context)))
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        context.artifacts["output_path"] = path


class PageSaving(StepGroup):
    """Save the page into the output directory under its source file name."""

    def __init__(self, options: OutputOptions) -> None:
        self.options = options

    def steps(self) -> list[Step]:
        return [SavePage(self.options)]
