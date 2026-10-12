"""Steps that translate the recognized text of regions from the source language with NLLB."""

import dataclasses
import logging
from collections.abc import Callable
from functools import cached_property
from pathlib import Path
from typing import cast

import numpy as np
import onnxruntime
from tokenizers import Tokenizer

from ..artifacts import get_text_regions
from ..config import Language, TranslationOptions
from ..context import PageContext
from ..step import Step, StepGroup

# NLLB codes of the supported languages.
_NLLB_CODES = {
    Language.EN: "eng_Latn",
    Language.PL: "pol_Latn",
    Language.DE: "deu_Latn",
    Language.FR: "fra_Latn",
    Language.ES: "spa_Latn",
    Language.IT: "ita_Latn",
    Language.PT: "por_Latn",
    Language.RU: "rus_Cyrl",
    Language.UK: "ukr_Cyrl",
    Language.ZH: "zho_Hans",
    Language.KO: "kor_Hang",
    Language.JA: "jpn_Jpan",
    Language.ID: "ind_Latn",
    Language.VI: "vie_Latn",
    Language.TR: "tur_Latn",
}
# Ends the source text, starts decoding and ends decoding.
_END_TOKEN = "</s>"
_WEIGHTS_URL = "https://huggingface.co/Xenova/nllb-200-distilled-600M/tree/main"

logger = logging.getLogger(__name__)


class NllbTranslator:
    """Runs an NLLB ONNX export on one text; the models are loaded on first use."""

    def __init__(self, options: TranslationOptions) -> None:
        self.options = options

    @cached_property
    def encoder(self) -> onnxruntime.InferenceSession:
        return _load_session(self.options.model_dir / "encoder_model.onnx")

    @cached_property
    def decoder(self) -> onnxruntime.InferenceSession:
        return _load_session(self.options.model_dir / "decoder_model.onnx")

    @cached_property
    def tokenizer(self) -> Tokenizer:
        return Tokenizer.from_file(str(_require(self.options.model_dir / "tokenizer.json")))

    def translate(self, text: str) -> str:
        end_id = self._token_id(_END_TOKEN)
        # The tokenizer's own template hardcodes English as the source, so build the ids by hand.
        source = [self._token_id(_NLLB_CODES[self.options.source_language])]
        source += self.tokenizer.encode(text, add_special_tokens=False).ids
        source.append(end_id)
        input_ids = np.array([source], dtype=np.int64)
        attention_mask = np.ones_like(input_ids)
        [features] = self.encoder.run(
            ["last_hidden_state"], {"input_ids": input_ids, "attention_mask": attention_mask}
        )

        def next_token_logits(ids: list[int]) -> np.ndarray:
            feeds = {
                "input_ids": np.array([ids], dtype=np.int64),
                "encoder_hidden_states": features,
                "encoder_attention_mask": attention_mask,
            }
            [logits] = cast(list[np.ndarray], self.decoder.run(["logits"], feeds))
            return logits[0, -1]

        prefix = [end_id, self._token_id(_NLLB_CODES[self.options.target_language])]
        ids = _greedy_decode(next_token_logits, prefix, end_id, self.options.max_length)
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def _token_id(self, token: str) -> int:
        token_id = self.tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f"Token {token} is missing from the translation tokenizer")
        return token_id


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"Translation model file not found at {path}; download it from {_WEIGHTS_URL}"
        )
    return path


def _load_session(path: Path) -> onnxruntime.InferenceSession:
    return onnxruntime.InferenceSession(_require(path), providers=["CPUExecutionProvider"])


def _greedy_decode(
    next_token_logits: Callable[[list[int]], np.ndarray],
    prefix: list[int],
    end_id: int,
    max_length: int,
) -> list[int]:
    """Extend the forced prefix with the most likely token until end_id or max_length tokens."""
    ids = list(prefix)
    while len(ids) < max_length and ids[-1] != end_id:
        ids.append(int(next_token_logits(ids).argmax()))
    return ids


class TranslateText:
    """Requires text_regions with original_text; replaces text_regions with translated_text filled in."""

    def __init__(self, translator: NllbTranslator) -> None:
        self.translator = translator

    async def execute(self, context: PageContext) -> None:
        regions = [
            dataclasses.replace(region, translated_text=self._translate(region.original_text))
            for region in get_text_regions(context)
        ]
        context.artifacts["text_regions"] = regions
        for region in regions:
            logger.info(
                "Text translated page=%s box=%s text=%s",
                context.source_path, region.box.as_tuple(), region.translated_text,
            )

    def _translate(self, text: str | None) -> str:
        if text is None:
            raise ValueError("Text regions must be recognized before they are translated")
        return self.translator.translate(text) if text else ""


class TextTranslation(StepGroup):
    """Translate the recognized text of every region into the target language."""

    def __init__(self, options: TranslationOptions) -> None:
        self.options = options

    def steps(self) -> list[Step]:
        return [TranslateText(NllbTranslator(self.options))]
