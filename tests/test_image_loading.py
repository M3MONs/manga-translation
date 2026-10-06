import tempfile
import unittest
from pathlib import Path

from PIL import Image

from manga_translation import ImageOptions, PageContext, Pipeline
from manga_translation.steps.image_loading import ImageLoading, ResizeImage

WHITE = (255, 255, 255)


class ImageLoadingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.directory = Path(temp_dir.name)

    async def load(self, image: Image.Image, name: str, max_size: int = 0, **save_options) -> Image.Image:
        path = self.directory / name
        image.save(path, **save_options)
        pipeline = Pipeline.from_groups([ImageLoading(ImageOptions(max_size))])
        context = await pipeline.run(PageContext(path))
        loaded = context.artifacts["image"]
        assert isinstance(loaded, Image.Image)
        return loaded

    async def test_transparent_images_become_rgb_on_white(self) -> None:
        palette = Image.new("P", (4, 4), 0)
        palette.putpalette([0, 0, 0] * 256)
        cases = {
            "palette.png": (palette, {"transparency": 0}),
            "rgba.png": (Image.new("RGBA", (4, 4), (255, 0, 0, 0)), {}),
            "la.png": (Image.new("LA", (4, 4), (0, 0)), {}),
        }
        for name, (image, save_options) in cases.items():
            with self.subTest(name):
                result = await self.load(image, name, **save_options)
                self.assertEqual(result.mode, "RGB")
                self.assertEqual(result.getpixel((0, 0)), WHITE)

    async def test_other_modes_become_rgb(self) -> None:
        cases = {
            "gray.jpg": Image.new("L", (4, 4), 128),
            "cmyk.jpg": Image.new("CMYK", (4, 4), (0, 0, 0, 0)),
            "gray16.png": Image.new("I;16", (4, 4), 65535),
        }
        for name, image in cases.items():
            with self.subTest(name):
                result = await self.load(image, name)
                self.assertEqual(result.mode, "RGB")

    async def test_16_bit_grayscale_keeps_full_range(self) -> None:
        result = await self.load(Image.new("I;16", (4, 4), 65535), "gray16.png")
        self.assertEqual(result.getpixel((0, 0)), WHITE)

    async def test_exif_orientation_is_applied(self) -> None:
        image = Image.new("RGB", (40, 20))
        exif = Image.Exif()
        exif[0x0112] = 6
        result = await self.load(image, "rotated.jpg", exif=exif)
        self.assertEqual(result.size, (20, 40))

    async def test_resize_only_downscales_large_images(self) -> None:
        small = await self.load(Image.new("RGB", (30, 20)), "small.png", max_size=40)
        self.assertEqual(small.size, (30, 20))
        large = await self.load(Image.new("RGB", (500, 200)), "large.png", max_size=40)
        self.assertEqual(large.size, (40, 16))

    def test_zero_max_size_disables_resize(self) -> None:
        steps = ImageLoading(ImageOptions(max_size=0)).steps()
        self.assertFalse(any(isinstance(step, ResizeImage) for step in steps))
