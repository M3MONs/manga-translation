"""Steps that read the original Japanese text of detected regions with manga-ocr."""

import dataclasses
import logging
import re
from collections.abc import Callable
from functools import cached_property
from pathlib import Path
from typing import cast

import numpy as np
import onnxruntime
from PIL import Image

from ..artifacts import get_image, get_text_regions
from ..config import Language, RecognitionOptions
from ..context import PageContext
from ..regions import TextRegion
from ..step import Step, StepGroup

_INPUT_SIZE = 224
_START_ID = 2
"""[CLS] starts decoding."""
_END_ID = 3
"""[SEP] ends decoding."""
_WEIGHTS_URL = "https://huggingface.co/dnouv/manga-ocr/tree/main"

logger = logging.getLogger(__name__)


class MangaOcr:
    """Runs a manga-ocr ONNX export on one text crop; the models are loaded on first use."""

    def __init__(self, options: RecognitionOptions) -> None:
        self.options = options

    @cached_property
    def encoder(self) -> onnxruntime.InferenceSession:
        return _load_session(self.options.model_dir / "encoder_model.onnx")

    @cached_property
    def decoder(self) -> onnxruntime.InferenceSession:
        return _load_session(self.options.model_dir / "decoder_model_merged.onnx")

    @cached_property
    def vocab(self) -> list[str]:
        return _require(self.options.model_dir / "vocab.txt").read_text("utf-8").splitlines()

    def recognize(self, crop: Image.Image) -> str:
        pixels = _to_model_input(crop)
        [features] = self.encoder.run(["last_hidden_state"], {"pixel_values": pixels})

        def next_token_logits(ids: list[int]) -> np.ndarray:
            tokens = np.array([ids], dtype=np.int64)
            feeds = {"input_ids": tokens, "encoder_hidden_states": features}
            [logits] = cast(list[np.ndarray], self.decoder.run(["logits"], feeds))
            return logits[0, -1]

        ids = _greedy_decode(next_token_logits, self.options.max_length)
        return _postprocess(_tokens_to_text(ids, self.vocab))


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"Text recognition model file not found at {path}; download it from {_WEIGHTS_URL}"
        )
    return path


def _load_session(path: Path) -> onnxruntime.InferenceSession:
    return onnxruntime.InferenceSession(_require(path), providers=["CPUExecutionProvider"])


def _to_model_input(crop: Image.Image) -> np.ndarray:
    """Grayscale crop stretched to 224x224 as a 3-channel NCHW float tensor in [-1, 1]."""
    gray = crop.convert("L").convert("RGB")
    image = gray.resize((_INPUT_SIZE, _INPUT_SIZE), Image.Resampling.BILINEAR)
    pixels = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)[np.newaxis] / 255
    return (pixels - 0.5) / 0.5


def _greedy_decode(
    next_token_logits: Callable[[list[int]], np.ndarray], max_length: int
) -> list[int]:
    """Pick the most likely token until the end token or max_length tokens, start token included."""
    ids = [_START_ID]
    while len(ids) < max_length and ids[-1] != _END_ID:
        ids.append(int(next_token_logits(ids).argmax()))
    return ids


def _tokens_to_text(ids: list[int], vocab: list[str]) -> str:
    """Join character tokens, skipping special ones such as [CLS], [SEP] and [UNK]."""
    tokens = (vocab[index] for index in ids)
    return "".join(token for token in tokens if not (token.startswith("[") and token.endswith("]")))


def _postprocess(text: str) -> str:
    """Clean up model output the way manga-ocr does."""
    text = "".join(text.split()).replace("…", "...")
    text = re.sub("[・.]{2,}", lambda dots: "." * len(dots.group()), text)
    return "".join(chr(ord(char) + 0xFEE0) if "!" <= char <= "~" else char for char in text)


class RecognizeText:
    """Requires image and text_regions; replaces text_regions with original_text filled in."""

    def __init__(self, ocr: MangaOcr) -> None:
        self.ocr = ocr

    async def execute(self, context: PageContext) -> None:
        image = get_image(context)
        regions = [
            dataclasses.replace(region, original_text=self._read(image, region))
            for region in get_text_regions(context)
        ]
        context.artifacts["text_regions"] = regions
        for region in regions:
            logger.info(
                "Text recognized page=%s box=%s text=%s",
                context.source_path, region.box.as_tuple(), region.original_text,
            )

    def _read(self, image: Image.Image, region: TextRegion) -> str:
        return self.ocr.recognize(image.crop(region.box.as_tuple()))


class TextRecognition(StepGroup):
    """Read the original text of every detected region."""

    def __init__(self, options: RecognitionOptions) -> None:
        self.options = options

    def steps(self) -> list[Step]:
        if self.options.source_language != Language.JA:
            raise ValueError(f"Text recognition for {self.options.source_language} is not supported yet")
        return [RecognizeText(MangaOcr(self.options))]
