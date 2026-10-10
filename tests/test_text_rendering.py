import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageFont

from manga_translation import Language, PageContext, Pipeline, RenderingOptions
from manga_translation.artifacts import get_output_image
from manga_translation.regions import BoundingBox, TextRegion
from manga_translation.steps.text_rendering import (
    FontLoader,
    TextRendering,
    _bubble_area,
    _fit_text,
    _glyph_size,
    _line_height,
    _outline_width,
    _render_area,
    _wrap,
)

WHITE = (255, 255, 255)
BOX = BoundingBox(20, 20, 180, 100)


def default_font(size: int) -> ImageFont.FreeTypeFont:
    return cast(ImageFont.FreeTypeFont, ImageFont.load_default(size))


class WrapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.font = default_font(20)

    def test_words_share_a_line_while_they_fit(self) -> None:
        width = round(self.font.getlength("one two"))
        self.assertEqual(_wrap("one two three", self.font, width), ["one two", "three"])

    def test_word_wider_than_width_is_broken_between_characters(self) -> None:
        width = round(self.font.getlength("abc"))
        self.assertEqual(_wrap("abcdef", self.font, width), ["abc", "def"])

    def test_lines_fit_the_width_without_breaking_words_that_fit(self) -> None:
        text = "a fairly long sentence with several words"
        width = round(max(self.font.getlength(word) for word in text.split()))
        lines = _wrap(text, self.font, width)
        self.assertTrue(all(self.font.getlength(line) <= width for line in lines))
        self.assertEqual(" ".join(lines), text)


class FitTextTests(unittest.TestCase):
    def test_short_text_uses_max_size(self) -> None:
        font, lines = _fit_text("Hi", (200, 100), default_font, 8, 30)
        self.assertEqual((font.size, lines), (30, ["Hi"]))

    def test_long_text_shrinks_until_it_fits(self) -> None:
        text = "this sentence is too long for the largest size"
        font, lines = _fit_text(text, (120, 60), default_font, 8, 30)
        self.assertLess(font.size, 30)
        self.assertLessEqual(len(lines) * _line_height(font), 60)

    def test_smaller_size_is_preferred_to_breaking_words(self) -> None:
        width = round(default_font(20).getlength("narrow"))
        font, lines = _fit_text("narrow column", (width, 400), default_font, 8, 30)
        self.assertLessEqual(font.size, 20)
        self.assertEqual(lines, ["narrow", "column"])

    def test_words_are_broken_when_no_size_fits_them_whole(self) -> None:
        _, lines = _fit_text("unbreakablylongword", (40, 400), default_font, 8, 30)
        self.assertEqual("".join(lines), "unbreakablylongword")
        self.assertGreater(len(lines), 1)

    def test_text_that_never_fits_uses_min_size(self) -> None:
        font, lines = _fit_text("word " * 50, (40, 20), default_font, 8, 30)
        self.assertEqual(font.size, 8)
        self.assertGreater(len(lines) * _line_height(font), 20)


class FontLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.latin = Path(temp_dir.name) / "latin.ttf"
        self.cjk = Path(temp_dir.name) / "cjk.otf"
        self.latin.write_bytes(b"latin")
        self.cjk.write_bytes(b"cjk")

    def loader(self, language: Language) -> FontLoader:
        return FontLoader(RenderingOptions(language, font_path=self.latin, cjk_font_path=self.cjk))

    def test_font_is_chosen_by_target_language(self) -> None:
        self.assertEqual(self.loader(Language.PL).data, b"latin")
        self.assertEqual(self.loader(Language.RU).data, b"latin")
        self.assertEqual(self.loader(Language.ZH).data, b"cjk")
        self.assertEqual(self.loader(Language.KO).data, b"cjk")

    def test_missing_font_is_reported(self) -> None:
        loader = FontLoader(RenderingOptions(font_path=Path("missing.ttf")))
        with self.assertRaisesRegex(FileNotFoundError, "missing.ttf"):
            loader.font(12)


class TextRenderingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        font = patch.object(
            FontLoader, "font", autospec=True, side_effect=lambda loader, size: default_font(size)
        )
        font.start()
        self.addCleanup(font.stop)

    async def render(
        self, translated_text: str | None, original_text: str = "テキスト"
    ) -> tuple[Image.Image, PageContext]:
        page = Image.new("RGB", (200, 120), WHITE)
        regions = [TextRegion(BOX, 0.9, original_text=original_text, translated_text=translated_text)]
        context = PageContext(Path("page.png"), {"output_image": page, "text_regions": regions})
        await Pipeline.from_groups([TextRendering(RenderingOptions())]).run(context)
        return page, context

    async def test_text_is_drawn_centered_on_the_box(self) -> None:
        _, context = await self.render("Hello there")
        output = get_output_image(context)
        dark = output.convert("L").point([255 if value < 128 else 0 for value in range(256)])
        bbox = dark.getbbox()
        assert bbox is not None
        self.assertAlmostEqual((bbox[0] + bbox[2]) / 2, (BOX.x1 + BOX.x2) / 2, delta=3)
        self.assertAlmostEqual((bbox[1] + bbox[3]) / 2, (BOX.y1 + BOX.y2) / 2, delta=6)

    async def test_font_is_not_larger_than_the_original_glyphs(self) -> None:
        # 32 original characters share the 160x80 box: about 20 px each.
        _, large = await self.render("Hi")
        _, capped = await self.render("Hi", original_text="テ" * 32)

        def text_height(context: PageContext) -> int:
            dark = get_output_image(context).convert("L").point([255 if v < 128 else 0 for v in range(256)])
            bbox = dark.getbbox()
            assert bbox is not None
            return bbox[3] - bbox[1]

        self.assertLess(text_height(capped), text_height(large))
        self.assertLessEqual(text_height(capped), 20)

    async def test_previous_output_image_is_not_modified(self) -> None:
        page, context = await self.render("Hello there")
        self.assertIsNot(get_output_image(context), page)
        self.assertEqual(page.getcolors(), [(200 * 120, WHITE)])

    async def test_empty_translation_draws_nothing(self) -> None:
        _, context = await self.render("")
        self.assertEqual(get_output_image(context).getcolors(), [(200 * 120, WHITE)])

    async def test_untranslated_region_is_an_error(self) -> None:
        with (
            self.assertLogs("manga_translation.pipeline", "ERROR"),
            self.assertRaisesRegex(ValueError, "translated before"),
        ):
            await self.render(None)


def blank_page(width: int = 400, height: int = 200) -> np.ndarray:
    return np.full((height, width), 255, dtype=np.uint8)


class RenderAreaTests(unittest.TestCase):
    def test_narrow_box_is_widened_around_its_center(self) -> None:
        # Grows 50 rows above and below, then widens to 0.8 of the new height.
        area = _render_area(blank_page(), BoundingBox(190, 50, 210, 150), [])
        self.assertEqual(area, BoundingBox(120, 0, 280, 200))

    def test_wide_box_only_grows_vertically(self) -> None:
        area = _render_area(blank_page(), BoundingBox(100, 50, 300, 100), [])
        self.assertEqual(area, BoundingBox(100, 25, 300, 125))

    def test_area_stops_before_regions_on_the_same_rows(self) -> None:
        others = [BoundingBox(170, 80, 180, 120), BoundingBox(0, 0, 400, 20)]
        area = _render_area(blank_page(), BoundingBox(190, 50, 210, 150), others)
        self.assertEqual(area, BoundingBox(184, 24, 261, 176))

    def test_area_stops_before_a_panel_border(self) -> None:
        gray = blank_page()
        gray[:, 220] = 0
        area = _render_area(gray, BoundingBox(190, 50, 210, 150), [])
        self.assertEqual(area, BoundingBox(120, 0, 214, 200))

    def test_dark_background_is_not_a_border(self) -> None:
        gray = np.zeros((200, 400), dtype=np.uint8)
        area = _render_area(gray, BoundingBox(190, 50, 210, 150), [])
        self.assertEqual(area, BoundingBox(120, 0, 280, 200))

    def test_area_is_clipped_to_the_page(self) -> None:
        area = _render_area(blank_page(), BoundingBox(390, 50, 400, 150), [])
        self.assertEqual(area, BoundingBox(315, 0, 400, 200))

    def test_area_grows_evenly_above_and_below(self) -> None:
        area = _render_area(blank_page(400, 400), BoundingBox(190, 150, 210, 250), [])
        self.assertEqual(area, BoundingBox(120, 100, 280, 300))

    def test_vertical_growth_stops_before_a_panel_border(self) -> None:
        gray = blank_page(400, 400)
        gray[130, :] = 0
        area = _render_area(gray, BoundingBox(190, 150, 210, 250), [])
        self.assertEqual((area.y1, area.y2), (137, 263))

    def test_vertical_growth_stops_before_regions_over_the_same_columns(self) -> None:
        others = [BoundingBox(180, 100, 220, 140)]
        area = _render_area(blank_page(400, 400), BoundingBox(190, 150, 210, 250), others)
        self.assertEqual((area.y1, area.y2), (144, 256))


class GlyphSizeTests(unittest.TestCase):
    def test_box_area_is_shared_by_the_characters(self) -> None:
        self.assertEqual(_glyph_size(BoundingBox(0, 0, 100, 100), "あいうえ"), 50)

    def test_outline_is_at_least_two_pixels(self) -> None:
        self.assertEqual(_outline_width(default_font(8)), 2)
        self.assertEqual(_outline_width(default_font(40)), 5)


def bubble() -> np.ndarray:
    """Light mask of a 400x300 page with a round bubble of radius 100 centered at (200, 150)."""
    ys, xs = np.mgrid[0:300, 0:400]
    return ((xs - 200) / 100) ** 2 + ((ys - 150) / 100) ** 2 <= 1


class BubbleAreaTests(unittest.TestCase):
    def test_area_grows_inside_the_bubble_only(self) -> None:
        light = bubble()
        area = _bubble_area(light, BoundingBox(190, 100, 210, 200), BoundingBox(0, 0, 400, 300))
        assert area is not None
        self.assertGreater(area.x2 - area.x1, 20)
        # Strips may hold 2% dark pixels, so only the very corners can touch the outline.
        self.assertGreaterEqual(light[area.y1 : area.y2, area.x1 : area.x2].mean(), 0.98)
        self.assertTrue(light[area.y1, (area.x1 + area.x2) // 2] and light[(area.y1 + area.y2) // 2, area.x1])

    def test_area_stays_within_bounds(self) -> None:
        bounds = BoundingBox(180, 90, 220, 210)
        area = _bubble_area(bubble(), BoundingBox(190, 100, 210, 200), bounds)
        self.assertEqual(area, bounds)

    def test_box_not_on_light_background_is_not_a_bubble(self) -> None:
        light = np.zeros((300, 400), dtype=bool)
        light[::2, ::2] = True  # screentone dots
        self.assertIsNone(_bubble_area(light, BoundingBox(190, 100, 210, 200), BoundingBox(0, 0, 400, 300)))
