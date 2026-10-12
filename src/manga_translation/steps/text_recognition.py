"""Steps that read the original text of detected regions: Japanese with manga-ocr, other
source languages with PaddleOCR."""

import dataclasses
import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import cached_property
from pathlib import Path
from typing import cast

import numpy as np
import onnxruntime
from PIL import Image

from ..artifacts import get_image, get_text_mask, get_text_regions
from ..config import Language, RecognitionOptions
from ..context import PageContext
from ..regions import TextRegion
from ..step import Step, StepGroup

_INPUT_SIZE = 224
# [CLS] starts decoding.
_START_ID = 2
# [SEP] ends decoding.
_END_ID = 3
_WEIGHTS_URL = "https://huggingface.co/dnouv/manga-ocr/tree/main"
_PADDLE_WEIGHTS_URL = "https://huggingface.co/monkt/paddleocr-onnx/tree/main/languages"
# Height in pixels of the text lines the PaddleOCR models read.
_PADDLE_HEIGHT = 48
# CTC blank, emitted between characters and never part of the text.
_BLANK_ID = 0
# Text joining the lines of one region, by source language read with PaddleOCR.
_LINE_SEPARATORS = {Language.EN: " ", Language.KO: " ", Language.ZH: ""}
# Text mask values above this (0-255) count as text when a region is split into lines.
_LINE_MASK_THRESHOLD = 64
# Rows of text pixels lower than this are noise, not a line of text.
_MIN_LINE_HEIGHT = 6
# Pixels added around each line, since the text mask hugs the glyphs.
_LINE_PADDING = 3

logger = logging.getLogger(__name__)


class TextRecognizer(ABC):
    """Reads the text of one region crop."""

    @abstractmethod
    def recognize(self, crop: Image.Image, mask: Image.Image) -> str:
        """Text in crop; mask is the text mask of the same area."""


class MangaOcr(TextRecognizer):
    """Runs a manga-ocr ONNX export on one text crop; the models are loaded on first use."""

    def __init__(self, options: RecognitionOptions) -> None:
        self.options = options

    @cached_property
    def encoder(self) -> onnxruntime.InferenceSession:
        return _load_session(self.options.manga_ocr_dir / "encoder_model.onnx", _WEIGHTS_URL)

    @cached_property
    def decoder(self) -> onnxruntime.InferenceSession:
        return _load_session(self.options.manga_ocr_dir / "decoder_model_merged.onnx", _WEIGHTS_URL)

    @cached_property
    def vocab(self) -> list[str]:
        return _require(self.options.manga_ocr_dir / "vocab.txt", _WEIGHTS_URL).read_text("utf-8").splitlines()

    def recognize(self, crop: Image.Image, mask: Image.Image) -> str:
        pixels = _to_model_input(crop)
        [features] = self.encoder.run(["last_hidden_state"], {"pixel_values": pixels})

        def next_token_logits(ids: list[int]) -> np.ndarray:
            tokens = np.array([ids], dtype=np.int64)
            feeds = {"input_ids": tokens, "encoder_hidden_states": features}
            [logits] = cast(list[np.ndarray], self.decoder.run(["logits"], feeds))
            return logits[0, -1]

        ids = _greedy_decode(next_token_logits, self.options.max_length)
        return _postprocess(_tokens_to_text(ids, self.vocab))


class PaddleOcr(TextRecognizer):
    """Runs a PaddleOCR recognition ONNX export (model.onnx, dict.txt in model_dir) on every line of
    horizontal text in a crop; the model is loaded on first use."""

    def __init__(self, model_dir: Path, line_separator: str) -> None:
        self.model_dir = model_dir
        self.line_separator = line_separator

    @cached_property
    def session(self) -> onnxruntime.InferenceSession:
        return _load_session(self.model_dir / "model.onnx", _PADDLE_WEIGHTS_URL)

    @cached_property
    def charset(self) -> list[str]:
        """Characters by class index: the blank, the dictionary, then the space."""
        characters = _require(self.model_dir / "dict.txt", _PADDLE_WEIGHTS_URL).read_text("utf-8")
        return ["", *characters.splitlines(), " "]

    def recognize(self, crop: Image.Image, mask: Image.Image) -> str:
        lines = _line_boxes(mask) or [(0, 0, crop.width, crop.height)]
        texts = [self._read_line(crop.crop(line)) for line in lines]
        return self.line_separator.join(text for text in texts if text)

    def _read_line(self, line: Image.Image) -> str:
        [probs] = cast(list[np.ndarray], self.session.run(None, {"x": _to_paddle_input(line)}))
        return _ctc_decode(probs[0], self.charset)


def _require(path: Path, url: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Text recognition model file not found at {path}; download it from {url}")
    return path


def _load_session(path: Path, url: str) -> onnxruntime.InferenceSession:
    return onnxruntime.InferenceSession(_require(path, url), providers=["CPUExecutionProvider"])


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


def _to_paddle_input(line: Image.Image) -> np.ndarray:
    """Line crop scaled to 48 pixels high, keeping its proportions, as an NCHW float tensor in
    [-1, 1]; channels are BGR because PaddleOCR reads its training images with OpenCV."""
    width = max(1, round(line.width * _PADDLE_HEIGHT / line.height))
    image = line.convert("RGB").resize((width, _PADDLE_HEIGHT), Image.Resampling.BILINEAR)
    pixels = np.asarray(image, dtype=np.float32)[:, :, ::-1].transpose(2, 0, 1)[np.newaxis] / 255
    return (pixels - 0.5) / 0.5


def _ctc_decode(probs: np.ndarray, charset: list[str]) -> str:
    """Most likely character at every step, with repeats merged and blanks dropped."""
    text = []
    previous = _BLANK_ID
    for index in probs.argmax(axis=1):
        if index != previous and index != _BLANK_ID:
            text.append(charset[index])
        previous = index
    return "".join(text)


def _line_boxes(mask: Image.Image) -> list[tuple[int, int, int, int]]:
    """Box (left, top, right, bottom; right and bottom exclusive) of every line of horizontal text
    in a text mask crop, padded by a few pixels. Lines are separated by rows without text pixels,
    and each box is trimmed to the columns with text, since the models invent text in blank space."""
    text = np.asarray(mask) > _LINE_MASK_THRESHOLD
    height, width = text.shape
    boxes = []
    for top, bottom in _text_row_runs(text.any(axis=1)):
        if bottom - top < _MIN_LINE_HEIGHT:
            continue
        columns = np.flatnonzero(text[top:bottom].any(axis=0))
        boxes.append((
            max(0, int(columns[0]) - _LINE_PADDING),
            max(0, top - _LINE_PADDING),
            min(width, int(columns[-1]) + 1 + _LINE_PADDING),
            min(height, bottom + _LINE_PADDING),
        ))
    return boxes


def _text_row_runs(rows: np.ndarray) -> list[tuple[int, int]]:
    """Top and bottom (exclusive) of every run of consecutive rows with text."""
    runs = []
    top = None
    for y, has_text in enumerate(rows):
        if has_text and top is None:
            top = y
        elif not has_text and top is not None:
            runs.append((top, y))
            top = None
    if top is not None:
        runs.append((top, len(rows)))
    return runs


class RecognizeText:
    """Requires image, text_mask and text_regions; replaces text_regions with original_text filled in."""

    def __init__(self, ocr: TextRecognizer) -> None:
        self.ocr = ocr

    async def execute(self, context: PageContext) -> None:
        image = get_image(context)
        mask = get_text_mask(context)
        regions = [
            dataclasses.replace(region, original_text=self._read(image, mask, region))
            for region in get_text_regions(context)
        ]
        context.artifacts["text_regions"] = regions
        for region in regions:
            logger.info(
                "Text recognized page=%s box=%s text=%s",
                context.source_path, region.box.as_tuple(), region.original_text,
            )

    def _read(self, image: Image.Image, mask: Image.Image, region: TextRegion) -> str:
        box = region.box.as_tuple()
        return self.ocr.recognize(image.crop(box), mask.crop(box))


class TextRecognition(StepGroup):
    """Read the original text of every detected region."""

    def __init__(self, options: RecognitionOptions) -> None:
        self.options = options

    def steps(self) -> list[Step]:
        return [RecognizeText(self._recognizer())]

    def _recognizer(self) -> TextRecognizer:
        """manga-ocr for Japanese, the PaddleOCR model of the language for the others."""
        language = self.options.source_language
        if language == Language.JA:
            return MangaOcr(self.options)
        return PaddleOcr(self.options.paddle_ocr_dir / language, _LINE_SEPARATORS[language])
