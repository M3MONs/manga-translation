"""Steps that erase the original text from a copy of the page with LaMa inpainting."""

from functools import cached_property
from typing import cast

import numpy as np
import onnxruntime
from PIL import Image, ImageFilter

from ..artifacts import get_image, get_text_mask, get_text_regions
from ..config import ErasingOptions
from ..context import PageContext
from ..regions import BoundingBox
from ..step import Step, StepGroup

_INPUT_SIZE = 512
_MARGIN_SCALE = 0.25
"""Margin around a region, relative to its shorter side, whose text is erased too;
detected boxes are tight and leave out furigana, punctuation and glyph outlines."""
_CONTEXT_SCALE = 1.25
"""Least side of the square inpainted around the erased area, relative to its longer side."""
_WEIGHTS_URL = "https://huggingface.co/Carve/LaMa-ONNX"


class LamaInpainter:
    """Runs LaMa ONNX weights on one image crop; the model is loaded on first use."""

    def __init__(self, options: ErasingOptions) -> None:
        self.options = options

    @cached_property
    def session(self) -> onnxruntime.InferenceSession:
        path = self.options.model_path
        if not path.is_file():
            raise FileNotFoundError(
                f"Inpainting weights not found at {path}; download lama_fp32.onnx from {_WEIGHTS_URL}"
            )
        return onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])

    def inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """Return image with its pixels under the non-zero mask pixels filled in, at image's size."""
        image_input, mask_input = _to_model_input(image, mask)
        # The model returns dense arrays; the stubs also allow sparse tensors, lists and dicts.
        [output] = cast(
            list[np.ndarray], self.session.run(["output"], {"image": image_input, "mask": mask_input})
        )
        return _from_model_output(output, image.size)


def _to_model_input(image: Image.Image, mask: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Scale image and mask to the model's fixed 512x512 NCHW float inputs: RGB in 0-1, mask 0 or 1."""
    size = (_INPUT_SIZE, _INPUT_SIZE)
    pixels = np.asarray(image.resize(size, Image.Resampling.BILINEAR), dtype=np.float32)
    image_input = pixels.transpose(2, 0, 1)[np.newaxis] / 255
    # Any covered pixel counts, so thin strokes do not vanish when the crop is scaled down.
    covered = np.asarray(mask.resize(size, Image.Resampling.BILINEAR)) > 0
    mask_input = covered.astype(np.float32)[np.newaxis, np.newaxis]
    return image_input, mask_input


def _from_model_output(output: np.ndarray, size: tuple[int, int]) -> Image.Image:
    """Turn the 0-255 NCHW output back into an RGB image of the given size."""
    pixels = output[0].transpose(1, 2, 0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(pixels, "RGB").resize(size, Image.Resampling.BILINEAR)


def _erase_area(box: BoundingBox, page_size: tuple[int, int]) -> BoundingBox:
    """The box grown by _MARGIN_SCALE times its shorter side on every edge, clipped to the page."""
    margin = round(_MARGIN_SCALE * min(box.x2 - box.x1, box.y2 - box.y1))
    return BoundingBox(
        max(0, box.x1 - margin),
        max(0, box.y1 - margin),
        min(page_size[0], box.x2 + margin),
        min(page_size[1], box.y2 + margin),
    )


def _context_window(box: BoundingBox, page_size: tuple[int, int]) -> BoundingBox:
    """Square centered on the box, shifted into the page; at least the model input size, so small
    areas are inpainted at full resolution, and _CONTEXT_SCALE times the box's longer side."""
    side = max(_INPUT_SIZE, round(_CONTEXT_SCALE * max(box.x2 - box.x1, box.y2 - box.y1)))
    starts = []
    for low, high, page_side in ((box.x1, box.x2, page_size[0]), (box.y1, box.y2, page_size[1])):
        start = (low + high - side) // 2
        starts.append(max(0, min(start, page_side - side)))
    x1, y1 = starts
    return BoundingBox(x1, y1, min(x1 + side, page_size[0]), min(y1 + side, page_size[1]))


class EraseText:
    """Requires image, text_mask and text_regions; produces output_image with the text of
    recognized regions inpainted away. The image artifact is unchanged."""

    def __init__(self, options: ErasingOptions, inpainter: LamaInpainter) -> None:
        self.options = options
        self.inpainter = inpainter

    async def execute(self, context: PageContext) -> None:
        threshold = self.options.mask_threshold
        mask = get_text_mask(context).point([255 if value > threshold else 0 for value in range(256)])
        # MaxFilter(1) would be a no-op, but Pillow crashes the process on it with SIGFPE.
        if self.options.mask_dilation > 1:
            mask = mask.filter(ImageFilter.MaxFilter(self.options.mask_dilation))
        output = get_image(context).copy()
        for region in get_text_regions(context):
            if not region.original_text:
                continue
            area = _erase_area(region.box, output.size)
            window = _context_window(area, output.size)
            # Only this region's text is replaced; text of other regions in the window stays.
            region_mask = Image.new("L", (window.x2 - window.x1, window.y2 - window.y1), 0)
            region_mask.paste(mask.crop(area.as_tuple()), (area.x1 - window.x1, area.y1 - window.y1))
            if region_mask.getbbox() is None:
                continue
            filled = self.inpainter.inpaint(output.crop(window.as_tuple()), region_mask)
            output.paste(filled, (window.x1, window.y1), region_mask)
        context.artifacts["output_image"] = output


class TextErasing(StepGroup):
    """Erase the original text of every recognized region on a copy of the page."""

    def __init__(self, options: ErasingOptions) -> None:
        self.options = options

    def steps(self) -> list[Step]:
        return [EraseText(self.options, LamaInpainter(self.options))]
