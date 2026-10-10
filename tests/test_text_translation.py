import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

import numpy as np
import onnxruntime
from tokenizers import AddedToken, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from manga_translation import Language, PageContext, Pipeline, TranslationOptions
from manga_translation.artifacts import get_text_regions
from manga_translation.regions import BoundingBox, TextRegion
from manga_translation.steps.text_translation import (
    _TARGET_CODES,
    NllbTranslator,
    TextTranslation,
    TranslateText,
    _greedy_decode,
)

REAL_TOKENIZER = TranslationOptions().model_dir / "tokenizer.json"
VOCAB = {"<unk>": 0, "</s>": 1, "jpn_Jpan": 2, "pol_Latn": 3, "こんにちは": 4, "cześć": 5, "świecie": 6}
END, JPN, POL, KONNICHIWA, CZESC, SWIECIE = 1, 2, 3, 4, 5, 6


def logits_for(token: int) -> np.ndarray:
    logits = np.zeros(len(VOCAB), dtype=np.float32)
    logits[token] = 1
    return logits


class DecodingTests(unittest.TestCase):
    def test_extends_prefix_until_end_token(self) -> None:
        script = iter([5, 6, END, 5])
        ids = _greedy_decode(lambda ids: logits_for(next(script)), [END, POL], END, max_length=10)
        self.assertEqual(ids, [END, POL, 5, 6, END])

    def test_stops_at_max_length(self) -> None:
        ids = _greedy_decode(lambda ids: logits_for(5), [END, POL], END, max_length=4)
        self.assertEqual(ids, [END, POL, 5, 5])

    def test_each_step_sees_tokens_so_far(self) -> None:
        seen = []

        def next_token_logits(ids: list[int]) -> np.ndarray:
            seen.append(list(ids))
            return logits_for(5 if len(ids) < 4 else END)

        _greedy_decode(next_token_logits, [END, POL], END, max_length=10)
        self.assertEqual(seen, [[END, POL], [END, POL, 5], [END, POL, 5, 5]])


class FakeSession:
    def __init__(self, outputs) -> None:
        self.outputs = outputs
        self.feeds: list[dict[str, np.ndarray]] = []

    def run(self, output_names, feeds):
        self.feeds.append(feeds)
        return [self.outputs(feeds)]


class NllbTranslatorTests(unittest.TestCase):
    def setUp(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        model_dir = Path(temp_dir.name)
        tokenizer = Tokenizer(WordLevel(VOCAB, unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.add_special_tokens([AddedToken(token) for token in ("</s>", "jpn_Jpan", "pol_Latn")])
        tokenizer.save(str(model_dir / "tokenizer.json"))
        self.translator = NllbTranslator(
            TranslationOptions(target_language=Language.PL, model_dir=model_dir)
        )
        self.features = np.ones((1, 3, 4), dtype=np.float32)
        self.encoder = FakeSession(lambda feeds: self.features)
        script = iter([CZESC, SWIECIE, END])
        self.decoder = FakeSession(lambda feeds: logits_for(next(script))[np.newaxis, np.newaxis])
        self.translator.encoder = cast(onnxruntime.InferenceSession, self.encoder)
        self.translator.decoder = cast(onnxruntime.InferenceSession, self.decoder)

    def test_source_is_japanese_text_between_language_code_and_end_token(self) -> None:
        self.translator.translate("こんにちは")
        [feeds] = self.encoder.feeds
        np.testing.assert_array_equal(feeds["input_ids"], [[JPN, KONNICHIWA, END]])
        np.testing.assert_array_equal(feeds["attention_mask"], [[1, 1, 1]])

    def test_decoding_starts_with_end_token_and_target_language(self) -> None:
        self.translator.translate("こんにちは")
        first = self.decoder.feeds[0]
        np.testing.assert_array_equal(first["input_ids"], [[END, POL]])
        self.assertIs(first["encoder_hidden_states"], self.features)
        np.testing.assert_array_equal(first["encoder_attention_mask"], [[1, 1, 1]])

    def test_result_skips_special_tokens(self) -> None:
        self.assertEqual(self.translator.translate("こんにちは"), "cześć świecie")

    def test_missing_weights_are_reported(self) -> None:
        translator = NllbTranslator(TranslationOptions(model_dir=Path("missing")))
        with self.assertRaisesRegex(FileNotFoundError, "tokenizer.json"):
            translator.translate("こんにちは")


class LanguageCodeTests(unittest.TestCase):
    def test_every_language_has_a_code(self) -> None:
        self.assertEqual(set(_TARGET_CODES), set(Language))

    @unittest.skipUnless(REAL_TOKENIZER.is_file(), "NLLB tokenizer not downloaded")
    def test_codes_exist_in_real_tokenizer(self) -> None:
        tokenizer = Tokenizer.from_file(str(REAL_TOKENIZER))
        for language, code in _TARGET_CODES.items():
            with self.subTest(language):
                self.assertIsNotNone(tokenizer.token_to_id(code))


class TextTranslationTests(unittest.IsolatedAsyncioTestCase):
    def context(self, *texts: str | None) -> PageContext:
        regions = [
            TextRegion(BoundingBox(0, 0, 10, 10 + index), 0.9, original_text=text)
            for index, text in enumerate(texts)
        ]
        return PageContext(Path("page.png"), {"text_regions": regions})

    async def translate(self, context: PageContext) -> list[str]:
        [step] = TextTranslation(TranslationOptions()).steps()
        assert isinstance(step, TranslateText)
        with patch.object(NllbTranslator, "translate", side_effect=lambda text: f"<{text}>") as fake:
            await Pipeline([step]).run(context)
        return [call.args[0] for call in fake.call_args_list]

    async def test_step_fills_translated_text(self) -> None:
        context = self.context("あ", "い")
        await self.translate(context)
        self.assertEqual(
            context.artifacts["text_regions"],
            [
                TextRegion(BoundingBox(0, 0, 10, 10), 0.9, original_text="あ", translated_text="<あ>"),
                TextRegion(BoundingBox(0, 0, 10, 11), 0.9, original_text="い", translated_text="<い>"),
            ],
        )

    async def test_empty_text_is_not_sent_to_the_model(self) -> None:
        context = self.context("", "い")
        self.assertEqual(await self.translate(context), ["い"])
        self.assertEqual(get_text_regions(context)[0].translated_text, "")

    async def test_unrecognized_regions_are_rejected(self) -> None:
        with (
            self.assertLogs("manga_translation.pipeline", "ERROR"),
            self.assertRaisesRegex(ValueError, "recognized before"),
        ):
            await self.translate(self.context(None))
