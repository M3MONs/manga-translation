"""Steps that draw the translated text of regions onto the page with erased text."""

import io
import math
from collections.abc import Callable
from functools import cached_property
from pathlib import Path

import numpy as np
from PIL import ImageDraw, ImageFont

from ..artifacts import get_output_image, get_text_regions
from ..config import Language, RenderingOptions
from ..context import PageContext
from ..regions import BoundingBox
from ..step import Step, StepGroup

_CJK_LANGUAGES = {Language.ZH, Language.KO}
"""Target languages whose script the Latin and Cyrillic font lacks."""
_FONTS_URL = "https://github.com/notofonts"
_TEXT_COLOR = "black"
_OUTLINE_COLOR = "white"
_AREA_ASPECT = 0.8
"""Least width of the area text is drawn in, relative to its region's height; detected boxes
follow vertical Japanese text and are too narrow for horizontal translations."""
_NEIGHBOR_GAP = 4
"""Pixels kept free between the drawing area and other regions on the same rows."""
_BORDER_GAP = 6
"""Pixels kept free between the drawing area and a panel border."""
_VERTICAL_GROWTH = 0.5
"""Most rows the drawing area grows by above and below its region, relative to the region's height."""
_LIGHT = 200
"""Gray level above which an erased pixel counts as the light inside of a speech bubble."""
_BUBBLE_SHARE = 0.9
"""Least share of light pixels in an erased region for it to count as lying in a bubble."""
_STRIP_SHARE = 0.98
"""Least share of light pixels in a strip the drawing area grows by inside a bubble."""

type Font = ImageFont.FreeTypeFont


class FontLoader:
    """Loads the font matching the target language; the file is read on first use and every size
    is loaded once."""

    def __init__(self, options: RenderingOptions) -> None:
        self.options = options
        self._fonts: dict[int, Font] = {}

    @cached_property
    def data(self) -> bytes:
        cjk = self.options.target_language in _CJK_LANGUAGES
        path = self.options.cjk_font_path if cjk else self.options.font_path
        return _require(path).read_bytes()

    def font(self, size: int) -> Font:
        if size not in self._fonts:
            self._fonts[size] = ImageFont.truetype(io.BytesIO(self.data), size)
        return self._fonts[size]


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Font not found at {path}; download {path.name} from {_FONTS_URL}")
    return path


def _wrap(text: str, font: Font, width: int) -> list[str]:
    """Break text into lines no wider than width, at spaces where possible; words wider than
    width (or text without spaces, such as Chinese) are broken between characters."""
    lines: list[str] = []
    for word in text.split():
        if lines and font.getlength(f"{lines[-1]} {word}") <= width:
            lines[-1] += f" {word}"
            continue
        lines.append("")
        for char in word:
            if lines[-1] and font.getlength(lines[-1] + char) > width:
                lines.append("")
            lines[-1] += char
    return lines


def _line_height(font: Font) -> int:
    ascent, descent = font.getmetrics()
    return ascent + descent


def _fit_text(
    text: str,
    size: tuple[int, int],
    load_font: Callable[[int], Font],
    min_size: int,
    max_size: int,
) -> tuple[Font, list[str]]:
    """Largest font size from max_size down whose wrapped lines fit in size, preferring sizes where
    no word is broken between characters; min_size when nothing fits."""
    width, height = size
    for allow_broken_words in (False, True):
        for font_size in range(max_size, min_size - 1, -1):
            font = load_font(font_size)
            if not allow_broken_words and any(font.getlength(word) > width for word in text.split()):
                continue
            lines = _wrap(text, font, width)
            fits_width = all(font.getlength(line) <= width for line in lines)
            if fits_width and len(lines) * _line_height(font) <= height:
                return font, lines
    font = load_font(min_size)
    return font, _wrap(text, font, width)


def _outline_width(font: Font) -> int:
    return max(2, round(font.size / 8))


def _glyph_size(box: BoundingBox, original_text: str) -> int:
    """Estimated size of the original glyphs: the box area shared evenly by its characters."""
    return round(math.sqrt((box.x2 - box.x1) * (box.y2 - box.y1) / len(original_text)))


def _is_border_line(line: np.ndarray, beside: np.ndarray, other_beside: np.ndarray) -> bool:
    """The line is dark while a parallel line 3 pixels away is light: a panel border next to the
    white gutter between panels. Dark backgrounds inside a panel have no light line beside them."""
    dark = (line < 90).mean() >= 0.9
    light = max((beside > 220).mean(), (other_beside > 220).mean()) >= 0.9
    return bool(dark and light)


def _is_border_column(gray: np.ndarray, x: int, y1: int, y2: int) -> bool:
    if x < 3 or x + 3 >= gray.shape[1]:
        return False
    return _is_border_line(gray[y1:y2, x], gray[y1:y2, x - 3], gray[y1:y2, x + 3])


def _is_border_row(gray: np.ndarray, y: int, x1: int, x2: int) -> bool:
    if y < 3 or y + 3 >= gray.shape[0]:
        return False
    return _is_border_line(gray[y, x1:x2], gray[y - 3, x1:x2], gray[y + 3, x1:x2])


def _vertical_room(gray: np.ndarray, box: BoundingBox, others: list[BoundingBox]) -> int:
    """Rows the box can grow by both above and below, so text stays centered on the original:
    at most _VERTICAL_GROWTH times its height, stopping before panel borders, other regions over
    the same columns and the page edges."""
    limit = round(_VERTICAL_GROWTH * (box.y2 - box.y1))
    above = 0
    while above < limit and box.y1 - above > 0:
        if _is_border_row(gray, box.y1 - above - 1, box.x1, box.x2):
            above = max(0, above - _BORDER_GAP)
            break
        above += 1
    below = 0
    while below < limit and box.y2 + below < gray.shape[0]:
        if _is_border_row(gray, box.y2 + below, box.x1, box.x2):
            below = max(0, below - _BORDER_GAP)
            break
        below += 1
    for other in others:
        if other.x1 < box.x2 and other.x2 > box.x1:
            if other.y2 <= box.y1:
                above = min(above, max(0, box.y1 - other.y2 - _NEIGHBOR_GAP))
            if other.y1 >= box.y2:
                below = min(below, max(0, other.y1 - box.y2 - _NEIGHBOR_GAP))
    return min(above, below)


def _render_area(gray: np.ndarray, box: BoundingBox, others: list[BoundingBox]) -> BoundingBox:
    """The box grown evenly above and below (see _vertical_room), then widened around its center to
    _AREA_ASPECT times its height, without reaching other regions on the same rows, panel borders
    or past the page; gray is the page in mode L."""
    room = _vertical_room(gray, box, others)
    box = BoundingBox(box.x1, box.y1 - room, box.x2, box.y2 + room)
    width, height = box.x2 - box.x1, box.y2 - box.y1
    extra = max(0, round(_AREA_ASPECT * height) - width) // 2
    x1, x2 = max(0, box.x1 - extra), min(gray.shape[1], box.x2 + extra)
    for other in others:
        if other.y1 < box.y2 and other.y2 > box.y1:
            if other.x2 <= box.x1:
                x1 = max(x1, other.x2 + _NEIGHBOR_GAP)
            if other.x1 >= box.x2:
                x2 = min(x2, other.x1 - _NEIGHBOR_GAP)
    for x in range(box.x1 - 1, x1 - 1, -1):
        if _is_border_column(gray, x, box.y1, box.y2):
            x1 = x + _BORDER_GAP
            break
    for x in range(box.x2, x2):
        if _is_border_column(gray, x, box.y1, box.y2):
            x2 = x - _BORDER_GAP
            break
    # Gaps can push the edges inside the box; the box itself is always available.
    return BoundingBox(min(x1, box.x1), box.y1, max(x2, box.x2), box.y2)


def _bubble_area(light: np.ndarray, box: BoundingBox, bounds: BoundingBox) -> BoundingBox | None:
    """For a box lying on a light background, such as a speech bubble, the box grown one pixel at a
    time on every side while the added strip stays light and inside bounds, so text stays within
    the bubble; None when the box is not mostly light (screentone, dark or drawn backgrounds)."""
    if light[box.y1 : box.y2, box.x1 : box.x2].mean() < _BUBBLE_SHARE:
        return None
    x1, y1, x2, y2 = box.as_tuple()
    grown = True
    while grown:
        grown = False
        if x1 > bounds.x1 and light[y1:y2, x1 - 1].mean() >= _STRIP_SHARE:
            x1, grown = x1 - 1, True
        if x2 < bounds.x2 and light[y1:y2, x2].mean() >= _STRIP_SHARE:
            x2, grown = x2 + 1, True
        if y1 > bounds.y1 and light[y1 - 1, x1:x2].mean() >= _STRIP_SHARE:
            y1, grown = y1 - 1, True
        if y2 < bounds.y2 and light[y2, x1:x2].mean() >= _STRIP_SHARE:
            y2, grown = y2 + 1, True
    return BoundingBox(x1, y1, x2, y2)


class RenderText:
    """Requires output_image and text_regions with translated_text; replaces output_image with a
    copy where every translation is drawn centered in an area grown from its region's box, kept
    inside the speech bubble when the region lies in one."""

    def __init__(self, options: RenderingOptions, fonts: FontLoader) -> None:
        self.options = options
        self.fonts = fonts

    async def execute(self, context: PageContext) -> None:
        output = get_output_image(context).copy()
        gray = np.asarray(output.convert("L"))
        light = gray > _LIGHT
        draw = ImageDraw.Draw(output)
        regions = get_text_regions(context)
        for region in regions:
            if region.translated_text is None:
                raise ValueError("Text regions must be translated before their text is rendered")
            if region.translated_text:
                others = [other.box for other in regions if other is not region]
                area = _render_area(gray, region.box, others)
                area = _bubble_area(light, region.box, area) or area
                max_size = self.options.max_font_size
                if region.original_text:
                    # Translations are not lettered larger than the original text.
                    max_size = min(max_size, _glyph_size(region.box, region.original_text))
                self._draw(draw, region.translated_text, area, max(max_size, self.options.min_font_size))
        context.artifacts["output_image"] = output

    def _draw(self, draw: ImageDraw.ImageDraw, text: str, box: BoundingBox, max_size: int) -> None:
        width, height = box.x2 - box.x1, box.y2 - box.y1
        font, lines = _fit_text(text, (width, height), self.fonts.font, self.options.min_font_size, max_size)
        line_height = _line_height(font)
        top = box.y1 + (height - len(lines) * line_height) / 2
        for index, line in enumerate(lines):
            left = box.x1 + (width - font.getlength(line)) / 2
            draw.text(
                (left, top + index * line_height),
                line,
                font=font,
                fill=_TEXT_COLOR,
                stroke_width=_outline_width(font),
                stroke_fill=_OUTLINE_COLOR,
            )


class TextRendering(StepGroup):
    """Draw the translated text of every region over the place of its original text."""

    def __init__(self, options: RenderingOptions) -> None:
        self.options = options

    def steps(self) -> list[Step]:
        return [RenderText(self.options, FontLoader(self.options))]
