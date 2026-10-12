import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

import numpy as np
import onnxruntime
from PIL import Image

from manga_translation import Language, PageContext, Pipeline, RecognitionOptions
from manga_translation.regions import BoundingBox, TextRegion
from manga_translation.steps.text_recognition import (
    MangaOcr,
    PaddleOcr,
    RecognizeText,
    TextRecognition,
    _ctc_decode,
    _greedy_decode,
    _line_boxes,
    _postprocess,
    _to_model_input,
    _to_paddle_input,
    _tokens_to_text,
)

VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "あ", "い"]


def logits_for(token: int) -> np.ndarray:
    logits = np.zeros(len(VOCAB), dtype=np.float32)
    logits[token] = 1
    return logits


class ModelInputTests(unittest.TestCase):
    def test_crop_becomes_normalized_gray_tensor(self) -> None:
        tensor = _to_model_input(Image.new("RGB", (40, 90), (255, 0, 0)))
        self.assertEqual(tensor.shape, (1, 3, 224, 224))
        self.assertEqual(tensor.dtype, np.float32)
        gray = 2 * 76 / 255 - 1  # PIL luma of pure red is 76
        np.testing.assert_allclose(tensor[0, :, 0, 0], [gray] * 3, rtol=1e-6)

    def test_values_span_minus_one_to_one(self) -> None:
        self.assertEqual(_to_model_input(Image.new("L", (5, 5), 0)).min(), -1)
        self.assertEqual(_to_model_input(Image.new("L", (5, 5), 255)).max(), 1)


class DecodingTests(unittest.TestCase):
    def test_stops_at_end_token(self) -> None:
        script = iter([5, 6, 3, 5])
        ids = _greedy_decode(lambda ids: logits_for(next(script)), max_length=10)
        self.assertEqual(ids, [2, 5, 6, 3])

    def test_stops_at_max_length(self) -> None:
        self.assertEqual(_greedy_decode(lambda ids: logits_for(5), max_length=3), [2, 5, 5])

    def test_each_step_sees_tokens_so_far(self) -> None:
        seen = []

        def next_token_logits(ids: list[int]) -> np.ndarray:
            seen.append(list(ids))
            return logits_for(5 if len(ids) < 3 else 3)

        _greedy_decode(next_token_logits, max_length=10)
        self.assertEqual(seen, [[2], [2, 5], [2, 5, 5]])

    def test_special_tokens_are_skipped(self) -> None:
        self.assertEqual(_tokens_to_text([2, 5, 1, 6, 3], VOCAB), "あい")


class PostprocessTests(unittest.TestCase):
    def test_whitespace_is_removed(self) -> None:
        self.assertEqual(_postprocess("あ い\nう"), "あいう")

    def test_dots_become_full_width(self) -> None:
        self.assertEqual(_postprocess("あ…"), "あ．．．")
        self.assertEqual(_postprocess("・・"), "．．")

    def test_ascii_becomes_full_width(self) -> None:
        self.assertEqual(_postprocess("A1!"), "Ａ１！")

    def test_japanese_is_unchanged(self) -> None:
        self.assertEqual(_postprocess("『錬金術』"), "『錬金術』")


# Blank, two characters and the space PaddleOCR appends to its dictionaries.
CHARSET = ["", "a", "b", " "]
BLANK, A, B, SPACE = 0, 1, 2, 3


def ctc_probs(*ids: int) -> np.ndarray:
    probs = np.zeros((len(ids), len(CHARSET)), dtype=np.float32)
    probs[np.arange(len(ids)), ids] = 1
    return probs


class PaddleInputTests(unittest.TestCase):
    def test_line_is_scaled_to_model_height_keeping_proportions(self) -> None:
        tensor = _to_paddle_input(Image.new("RGB", (100, 20)))
        self.assertEqual(tensor.shape, (1, 3, 48, 240))
        self.assertEqual(tensor.dtype, np.float32)

    def test_values_span_minus_one_to_one(self) -> None:
        self.assertEqual(_to_paddle_input(Image.new("L", (5, 5), 0)).min(), -1)
        self.assertEqual(_to_paddle_input(Image.new("L", (5, 5), 255)).max(), 1)

    def test_channels_are_bgr(self) -> None:
        tensor = _to_paddle_input(Image.new("RGB", (5, 5), (255, 0, 0)))
        np.testing.assert_array_equal(tensor[0, :, 0, 0], [-1, -1, 1])


class CtcDecodingTests(unittest.TestCase):
    def test_repeats_are_merged_unless_separated_by_blank(self) -> None:
        self.assertEqual(_ctc_decode(ctc_probs(BLANK, A, A, BLANK, A, B), CHARSET), "aab")

    def test_only_blanks_give_empty_text(self) -> None:
        self.assertEqual(_ctc_decode(ctc_probs(BLANK, BLANK), CHARSET), "")

    def test_last_class_is_space(self) -> None:
        self.assertEqual(_ctc_decode(ctc_probs(A, SPACE, B), CHARSET), "a b")


class MangaOcrTests(unittest.TestCase):
    def test_missing_weights_are_reported(self) -> None:
        ocr = MangaOcr(RecognitionOptions(manga_ocr_dir=Path("missing")))
        with self.assertRaisesRegex(FileNotFoundError, "encoder_model.onnx"):
            ocr.recognize(Image.new("RGB", (10, 10)), Image.new("L", (10, 10)))


def line_mask(width: int, height: int, *lines: tuple[int, int, int, int]) -> Image.Image:
    """Text mask with text on every (left, top, right, bottom) box."""
    mask = Image.new("L", (width, height))
    for box in lines:
        mask.paste(255, box)
    return mask


class LineBoxesTests(unittest.TestCase):
    def test_lines_are_split_at_rows_without_text_and_padded(self) -> None:
        mask = line_mask(100, 60, (10, 10, 90, 20), (10, 35, 90, 50))
        self.assertEqual(_line_boxes(mask), [(7, 7, 93, 23), (7, 32, 93, 53)])

    def test_lines_are_trimmed_to_columns_with_text(self) -> None:
        mask = line_mask(100, 60, (10, 10, 90, 20), (40, 35, 60, 50))
        self.assertEqual(_line_boxes(mask), [(7, 7, 93, 23), (37, 32, 63, 53)])

    def test_mask_without_text_has_no_lines(self) -> None:
        self.assertEqual(_line_boxes(line_mask(100, 60)), [])

    def test_lines_at_edges_end_at_edges(self) -> None:
        mask = line_mask(100, 60, (0, 0, 100, 10), (0, 50, 100, 60))
        self.assertEqual(_line_boxes(mask), [(0, 0, 100, 13), (0, 47, 100, 60)])

    def test_weak_mask_values_are_not_text(self) -> None:
        mask = line_mask(100, 60, (10, 10, 90, 20))
        mask.paste(64, (0, 30, 100, 50))
        self.assertEqual(_line_boxes(mask), [(7, 7, 93, 23)])

    def test_rows_lower_than_a_line_are_noise(self) -> None:
        mask = line_mask(100, 60, (10, 10, 90, 15), (10, 30, 90, 40))
        self.assertEqual(_line_boxes(mask), [(7, 27, 93, 43)])


class FakeSession:
    """Returns the given outputs one per run and remembers the feeds."""

    def __init__(self, *outputs: np.ndarray) -> None:
        self.outputs = iter(outputs)
        self.feeds: list[dict[str, np.ndarray]] = []

    def run(self, output_names, feeds):
        self.feeds.append(feeds)
        return [next(self.outputs)]


class PaddleOcrTests(unittest.TestCase):
    def setUp(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.model_dir = Path(temp_dir.name)
        (self.model_dir / "dict.txt").write_text("a\nb\n", "utf-8")

    def ocr(self, line_separator: str, *outputs: np.ndarray) -> tuple[PaddleOcr, FakeSession]:
        ocr = PaddleOcr(self.model_dir, line_separator)
        session = FakeSession(*outputs)
        ocr.session = cast(onnxruntime.InferenceSession, session)
        return ocr, session

    def test_every_line_is_read_and_joined_with_separator(self) -> None:
        ocr, session = self.ocr(" ", ctc_probs(A, A, BLANK, A)[np.newaxis], ctc_probs(B)[np.newaxis])
        text = ocr.recognize(Image.new("RGB", (100, 60)), line_mask(100, 60, (0, 5, 100, 15), (0, 30, 100, 50)))
        self.assertEqual(text, "aa b")
        heights = [feeds["x"].shape[2] for feeds in session.feeds]
        widths = [feeds["x"].shape[3] for feeds in session.feeds]
        self.assertEqual(heights, [48, 48])
        self.assertEqual(widths, [round(100 * 48 / 16), round(100 * 48 / 26)])

    def test_lines_can_be_joined_without_separator(self) -> None:
        ocr, _ = self.ocr("", ctc_probs(A)[np.newaxis], ctc_probs(B)[np.newaxis])
        text = ocr.recognize(Image.new("RGB", (100, 60)), line_mask(100, 60, (0, 5, 100, 15), (0, 30, 100, 50)))
        self.assertEqual(text, "ab")

    def test_crop_without_mask_lines_is_read_as_one_line(self) -> None:
        ocr, session = self.ocr(" ", ctc_probs(A, SPACE, B)[np.newaxis])
        self.assertEqual(ocr.recognize(Image.new("RGB", (100, 20)), line_mask(100, 20)), "a b")
        [feeds] = session.feeds
        self.assertEqual(feeds["x"].shape, (1, 3, 48, 240))

    def test_empty_lines_are_skipped(self) -> None:
        ocr, _ = self.ocr(" ", ctc_probs(BLANK)[np.newaxis], ctc_probs(B)[np.newaxis])
        text = ocr.recognize(Image.new("RGB", (100, 60)), line_mask(100, 60, (0, 5, 100, 15), (0, 30, 100, 50)))
        self.assertEqual(text, "b")

    def test_charset_wraps_dictionary_in_blank_and_space(self) -> None:
        self.assertEqual(PaddleOcr(self.model_dir, " ").charset, CHARSET)

    def test_missing_weights_are_reported(self) -> None:
        ocr = PaddleOcr(self.model_dir / "missing", " ")
        with self.assertRaisesRegex(FileNotFoundError, "model.onnx.*paddleocr-onnx"):
            ocr.recognize(Image.new("RGB", (10, 10)), Image.new("L", (10, 10)))


class TextRecognitionGroupTests(unittest.TestCase):
    def test_japanese_text_is_read_with_manga_ocr(self) -> None:
        [step] = TextRecognition(RecognitionOptions(source_language=Language.JA)).steps()
        assert isinstance(step, RecognizeText)
        self.assertIsInstance(step.ocr, MangaOcr)

    def test_other_languages_are_read_with_their_paddle_ocr_model(self) -> None:
        expected = {Language.EN: " ", Language.KO: " ", Language.ZH: ""}
        for language, separator in expected.items():
            with self.subTest(language):
                options = RecognitionOptions(source_language=language, paddle_ocr_dir=Path("paddle"))
                [step] = TextRecognition(options).steps()
                assert isinstance(step, RecognizeText)
                assert isinstance(step.ocr, PaddleOcr)
                self.assertEqual(step.ocr.model_dir, Path("paddle") / language)
                self.assertEqual(step.ocr.line_separator, separator)


class TextRecognitionTests(unittest.IsolatedAsyncioTestCase):
    async def test_step_fills_original_text_from_region_crops(self) -> None:
        [step] = TextRecognition(RecognitionOptions()).steps()
        assert isinstance(step, RecognizeText)
        regions = [
            TextRegion(BoundingBox(0, 0, 10, 20), 0.9),
            TextRegion(BoundingBox(5, 5, 35, 15), 0.8),
        ]
        mask = Image.new("L", (40, 40))
        mask.paste(255, (5, 5, 35, 15))
        context = PageContext(
            Path("page.png"),
            {"image": Image.new("RGB", (40, 40)), "text_mask": mask, "text_regions": regions},
        )

        def describe(crop: Image.Image, mask: Image.Image) -> str:
            return f"{crop.width}x{crop.height} mask {mask.width}x{mask.height} {mask.getpixel((0, 0))}"

        with patch.object(MangaOcr, "recognize", side_effect=describe):
            await Pipeline([step]).run(context)
        self.assertEqual(
            context.artifacts["text_regions"],
            [
                TextRegion(BoundingBox(0, 0, 10, 20), 0.9, original_text="10x20 mask 10x20 0"),
                TextRegion(BoundingBox(5, 5, 35, 15), 0.8, original_text="30x10 mask 30x10 255"),
            ],
        )
