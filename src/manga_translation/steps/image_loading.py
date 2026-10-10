"""Steps that turn a source file into a uniform RGB image artifact."""

from PIL import Image, ImageOps

from ..artifacts import get_image
from ..config import ImageOptions
from ..context import PageContext
from ..step import Step, StepGroup

_ALPHA_MODES = {"RGBA", "RGBa", "LA", "La", "PA"}


class LoadImage:
    """Requires source_path; produces the image artifact."""

    async def execute(self, context: PageContext) -> None:
        with Image.open(context.source_path) as image:
            image.load()
        context.artifacts["image"] = image


class NormalizeImage:
    """Requires image; replaces it with an image without EXIF rotation, palette, or alpha."""

    async def execute(self, context: PageContext) -> None:
        image = ImageOps.exif_transpose(get_image(context))
        if image.mode == "P" or "transparency" in image.info:
            image = image.convert("RGBA" if "transparency" in image.info else "RGB")
        elif image.mode.startswith("I"):
            image = image.convert("I").point(lambda value: value / 256).convert("L")
        if image.mode in _ALPHA_MODES:
            background = Image.new("RGBA", image.size, "white")
            image = Image.alpha_composite(background, image.convert("RGBA")).convert("RGB")
        context.artifacts["image"] = image


class ConvertToRGB:
    """Requires image; replaces it with its RGB version."""

    async def execute(self, context: PageContext) -> None:
        image = get_image(context)
        if image.mode != "RGB":
            context.artifacts["image"] = image.convert("RGB")


class ResizeImage:
    """Requires image; downscales it when its longest side exceeds max_size."""

    def __init__(self, max_size: int) -> None:
        self.max_size = max_size

    async def execute(self, context: PageContext) -> None:
        image = get_image(context)
        if max(image.size) > self.max_size:
            context.artifacts["image"] = ImageOps.contain(
                image, (self.max_size, self.max_size), Image.Resampling.LANCZOS
            )


class ImageLoading(StepGroup):
    """Load → normalize → RGB → optional resize (skipped when max_size is 0)."""

    def __init__(self, options: ImageOptions) -> None:
        self.options = options

    def steps(self) -> list[Step]:
        steps: list[Step] = [LoadImage(), NormalizeImage(), ConvertToRGB()]
        if self.options.max_size > 0:
            steps.append(ResizeImage(self.options.max_size))
        return steps
