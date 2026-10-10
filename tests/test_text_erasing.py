import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from manga_translation import ErasingOptions, PageContext, Pipeline
from manga_translation.artifacts import get_image, get_output_image
from manga_translation.regions import BoundingBox, TextRegion
from manga_translation.steps.text_erasing import (
    LamaInpainter,
    TextErasing,
    _context_window,
    _erase_area,
    _from_model_output,
    _to_model_input,
)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
BOX = BoundingBox(10, 10, 30, 30)
NO_DILATION = ErasingOptions(mask_dilation=1)


def page() -> Image.Image:
    """Gray page with a white bubble at BOX and black "text" pixels inside it."""
    image = Image.new("RGB", (40, 40), GRAY)
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 29, 29), fill=WHITE)
    draw.rectangle((15, 15, 24, 24), fill=BLACK)
    return image


def text_mask(value: int = 255) -> Image.Image:
    mask = Image.new("L", (40, 40), 0)
    ImageDraw.Draw(mask).rectangle((15, 15, 24, 24), fill=value)
    return mask


def fill_white(image: Image.Image, mask: Image.Image) -> Image.Image:
    return Image.new("RGB", image.size, WHITE)


class FakeSession:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output
        self.feeds: dict[str, np.ndarray] = {}

    def run(self, output_names: list[str], feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.feeds = feeds
        return [self.output]


class LamaInpainterTests(unittest.TestCase):
    def test_inputs_are_scaled_to_model_size(self) -> None:
        mask = Image.new("L", (100, 50), 0)
        mask.putpixel((0, 0), 255)
        image_input, mask_input = _to_model_input(Image.new("RGB", (100, 50), (255, 0, 0)), mask)
        self.assertEqual(image_input.shape, (1, 3, 512, 512))
        self.assertEqual(image_input[0, :, 0, 0].tolist(), [1, 0, 0])
        self.assertEqual(mask_input.shape, (1, 1, 512, 512))
        self.assertEqual(sorted(np.unique(mask_input).tolist()), [0, 1])
        self.assertEqual(mask_input[0, 0, 0, 0], 1)

    def test_output_is_clipped_and_resized(self) -> None:
        output = np.full((1, 3, 512, 512), 300, dtype=np.float32)
        image = _from_model_output(output, (100, 50))
        self.assertEqual((image.mode, image.size), ("RGB", (100, 50)))
        self.assertEqual(image.getpixel((0, 0)), WHITE)

    def test_inpaint_runs_session_and_returns_crop_size(self) -> None:
        inpainter = LamaInpainter(ErasingOptions())
        session = FakeSession(np.zeros((1, 3, 512, 512), dtype=np.float32))
        inpainter.__dict__["session"] = session
        result = inpainter.inpaint(Image.new("RGB", (60, 30)), Image.new("L", (60, 30)))
        self.assertEqual(result.size, (60, 30))
        self.assertEqual(set(session.feeds), {"image", "mask"})

    def test_missing_weights_are_reported(self) -> None:
        inpainter = LamaInpainter(ErasingOptions(model_path=Path("missing.onnx")))
        with self.assertRaisesRegex(FileNotFoundError, "lama_fp32.onnx"):
            inpainter.inpaint(Image.new("RGB", (10, 10)), Image.new("L", (10, 10)))


class EraseAreaTests(unittest.TestCase):
    def test_box_grows_by_quarter_of_shorter_side(self) -> None:
        self.assertEqual(_erase_area(BoundingBox(10, 10, 30, 50), (100, 100)), BoundingBox(5, 5, 35, 55))

    def test_area_is_clipped_to_the_page(self) -> None:
        self.assertEqual(_erase_area(BoundingBox(0, 0, 40, 40), (45, 45)), BoundingBox(0, 0, 45, 45))


class ContextWindowTests(unittest.TestCase):
    def test_small_box_gets_model_size_window_at_full_resolution(self) -> None:
        window = _context_window(BoundingBox(990, 990, 1010, 1010), (2000, 2000))
        self.assertEqual(window, BoundingBox(744, 744, 1256, 1256))

    def test_large_box_gets_window_scaled_from_longer_side(self) -> None:
        window = _context_window(BoundingBox(0, 0, 100, 800), (2000, 2000))
        self.assertEqual(window, BoundingBox(0, 0, 1000, 1000))

    def test_window_is_shifted_into_the_page(self) -> None:
        window = _context_window(BoundingBox(1990, 0, 2000, 10), (2000, 2000))
        self.assertEqual(window, BoundingBox(1488, 0, 2000, 512))

    def test_window_is_clipped_to_a_smaller_page(self) -> None:
        window = _context_window(BoundingBox(0, 0, 30, 10), (40, 20))
        self.assertEqual(window, BoundingBox(0, 0, 40, 20))


class TextErasingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        inpaint = patch.object(
            LamaInpainter,
            "inpaint",
            autospec=True,
            side_effect=lambda inpainter, image, mask: fill_white(image, mask),
        )
        self.inpaint = inpaint.start()
        self.addCleanup(inpaint.stop)

    async def erase(
        self,
        regions: list[TextRegion],
        image: Image.Image | None = None,
        mask: Image.Image | None = None,
        options: ErasingOptions = NO_DILATION,
    ) -> PageContext:
        context = PageContext(
            Path("page.png"),
            {"image": image or page(), "text_mask": mask or text_mask(), "text_regions": regions},
        )
        return await Pipeline.from_groups([TextErasing(options)]).run(context)

    async def test_text_pixels_are_inpainted(self) -> None:
        context = await self.erase([TextRegion(BOX, 0.9, original_text="テキスト")])
        output = get_output_image(context)
        for point in [(15, 15), (20, 20), (24, 24)]:
            self.assertEqual(output.getpixel(point), WHITE)

    async def test_pixels_outside_region_mask_are_unchanged(self) -> None:
        mask = Image.new("L", (40, 40), 255)
        context = await self.erase([TextRegion(BOX, 0.9, original_text="テキスト")], mask=mask)
        output = get_output_image(context)
        self.assertEqual(output.getpixel((2, 2)), GRAY)
        self.assertEqual(output.getpixel((37, 37)), GRAY)

    async def test_image_artifact_is_unchanged(self) -> None:
        context = await self.erase([TextRegion(BOX, 0.9, original_text="テキスト")])
        self.assertEqual(get_image(context).getpixel((20, 20)), BLACK)

    async def test_regions_without_text_are_kept(self) -> None:
        for original_text in ("", None):
            with self.subTest(original_text):
                context = await self.erase([TextRegion(BOX, 0.9, original_text=original_text)])
                self.assertEqual(get_output_image(context).getpixel((20, 20)), BLACK)
        self.inpaint.assert_not_called()

    async def test_mask_below_threshold_is_not_text(self) -> None:
        regions = [TextRegion(BOX, 0.9, original_text="テキスト")]
        options = ErasingOptions(mask_threshold=100, mask_dilation=1)
        context = await self.erase(regions, mask=text_mask(100), options=options)
        self.assertEqual(get_output_image(context).getpixel((20, 20)), BLACK)
        self.inpaint.assert_not_called()

    async def test_dilation_grows_mask(self) -> None:
        image = page()
        ImageDraw.Draw(image).point((14, 20), fill=BLACK)
        regions = [TextRegion(BOX, 0.9, original_text="テキスト")]
        context = await self.erase(regions, image=image, options=ErasingOptions(mask_dilation=3))
        self.assertEqual(get_output_image(context).getpixel((14, 20)), WHITE)

    async def test_text_just_outside_the_box_is_erased(self) -> None:
        image = page()
        ImageDraw.Draw(image).point((8, 20), fill=BLACK)
        mask = text_mask()
        mask.putpixel((8, 20), 255)
        regions = [TextRegion(BOX, 0.9, original_text="テキスト")]
        context = await self.erase(regions, image=image, mask=mask)
        self.assertEqual(get_output_image(context).getpixel((8, 20)), WHITE)

    async def test_inpainter_gets_context_window_and_region_mask(self) -> None:
        await self.erase([TextRegion(BOX, 0.9, original_text="テキスト")])
        [call] = self.inpaint.call_args_list
        _, crop, mask = call.args
        self.assertEqual(crop.size, (40, 40))
        self.assertEqual(mask.getbbox(), (15, 15, 25, 25))
