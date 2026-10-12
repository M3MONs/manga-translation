import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from manga_translation import Language, PageContext, Pipeline, RecognitionOptions
from manga_translation.regions import BoundingBox, TextRegion
from manga_translation.steps.text_recognition import (
    MangaOcr,
    RecognizeText,
    TextRecognition,
    _greedy_decode,
    _postprocess,
    _to_model_input,
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


class MangaOcrTests(unittest.TestCase):
    def test_missing_weights_are_reported(self) -> None:
        ocr = MangaOcr(RecognitionOptions(model_dir=Path("missing")))
        with self.assertRaisesRegex(FileNotFoundError, "encoder_model.onnx"):
            ocr.recognize(Image.new("RGB", (10, 10)))


class TextRecognitionGroupTests(unittest.TestCase):
    def test_japanese_text_is_read_with_manga_ocr(self) -> None:
        [step] = TextRecognition(RecognitionOptions(source_language=Language.JA)).steps()
        assert isinstance(step, RecognizeText)
        self.assertIsInstance(step.ocr, MangaOcr)

    def test_other_source_languages_are_not_supported_yet(self) -> None:
        for language in (Language.EN, Language.KO, Language.ZH):
            with self.subTest(language), self.assertRaisesRegex(ValueError, "not supported yet"):
                TextRecognition(RecognitionOptions(source_language=language)).steps()


class TextRecognitionTests(unittest.IsolatedAsyncioTestCase):
    async def test_step_fills_original_text_from_region_crops(self) -> None:
        [step] = TextRecognition(RecognitionOptions()).steps()
        assert isinstance(step, RecognizeText)
        regions = [
            TextRegion(BoundingBox(0, 0, 10, 20), 0.9),
            TextRegion(BoundingBox(5, 5, 35, 15), 0.8),
        ]
        context = PageContext(
            Path("page.png"),
            {"image": Image.new("RGB", (40, 40)), "text_regions": regions},
        )
        with patch.object(
            MangaOcr, "recognize", side_effect=lambda crop: f"{crop.width}x{crop.height}"
        ):
            await Pipeline([step]).run(context)
        self.assertEqual(
            context.artifacts["text_regions"],
            [
                TextRegion(BoundingBox(0, 0, 10, 20), 0.9, original_text="10x20"),
                TextRegion(BoundingBox(5, 5, 35, 15), 0.8, original_text="30x10"),
            ],
        )
