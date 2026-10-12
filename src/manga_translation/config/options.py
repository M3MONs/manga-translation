"""Application parameters grouped into immutable parameter objects."""

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class Language(StrEnum):
    """Target language of the translation, as an ISO 639-1 code."""

    EN = "en"
    PL = "pl"
    DE = "de"
    FR = "fr"
    ES = "es"
    IT = "it"
    PT = "pt"
    RU = "ru"
    UK = "uk"
    ZH = "zh"
    KO = "ko"
    JA = "ja"
    ID = "id"
    VI = "vi"
    TR = "tr"


# Languages the text on the pages can be read and translated from.
SOURCE_LANGUAGES = frozenset({Language.JA, Language.EN, Language.KO, Language.ZH})


def _check_source_language(language: Language) -> None:
    if language not in SOURCE_LANGUAGES:
        raise ValueError(f"source_language must be one of {', '.join(sorted(SOURCE_LANGUAGES))}")


@dataclass(frozen=True)
class ImageOptions:
    # Longest allowed image side in pixels; 0 disables resizing.
    max_size: int = 4096

    def __post_init__(self) -> None:
        if self.max_size < 0:
            raise ValueError("max_size must be >= 0")


@dataclass(frozen=True)
class DetectionOptions:
    # comic-text-detector ONNX weights, downloaded by `download-models`.
    model_path: Path = Path("models/comictextdetector.pt.onnx")
    # Detections scoring below this are discarded.
    confidence_threshold: float = 0.4
    # IoU above which overlapping detections are merged into the best one.
    nms_threshold: float = 0.35

    def __post_init__(self) -> None:
        for name in ("confidence_threshold", "nms_threshold"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True)
class RecognitionOptions:
    # Language of the text on the pages; one of SOURCE_LANGUAGES.
    source_language: Language = Language.JA
    # manga-ocr ONNX export (encoder, merged decoder, vocab.txt) reading Japanese, downloaded by `download-models`.
    manga_ocr_dir: Path = Path("models/manga-ocr")
    # PaddleOCR recognition exports reading the other source languages, one directory per language
    # code (en, ko, zh) holding model.onnx and dict.txt, downloaded by `download-models`.
    paddle_ocr_dir: Path = Path("models/paddleocr")
    # Most tokens manga-ocr decodes per text region, including the start token.
    max_length: int = 300

    def __post_init__(self) -> None:
        _check_source_language(self.source_language)
        if self.max_length <= 0:
            raise ValueError("max_length must be > 0")


@dataclass(frozen=True)
class TranslationOptions:
    target_language: Language = Language.EN
    # Language of the text on the pages; one of SOURCE_LANGUAGES.
    source_language: Language = Language.JA
    # NLLB ONNX export (encoder, decoder, tokenizer.json), downloaded by `download-models`.
    model_dir: Path = Path("models/nllb-200-distilled-600M")
    # Most tokens decoded per text region, including the forced prefix.
    max_length: int = 256

    def __post_init__(self) -> None:
        _check_source_language(self.source_language)
        if self.max_length <= 0:
            raise ValueError("max_length must be > 0")


@dataclass(frozen=True)
class ErasingOptions:
    # LaMa inpainting ONNX weights, downloaded by `download-models`.
    model_path: Path = Path("models/lama/lama_fp32.onnx")
    # Text mask values above this (0–255) count as text pixels.
    mask_threshold: int = 64
    # Odd size of the filter growing the text mask to cover glyph outlines; 1 disables it.
    mask_dilation: int = 21

    def __post_init__(self) -> None:
        if not 0 <= self.mask_threshold <= 255:
            raise ValueError("mask_threshold must be between 0 and 255")
        if self.mask_dilation < 1 or self.mask_dilation % 2 == 0:
            raise ValueError("mask_dilation must be an odd number >= 1")


@dataclass(frozen=True)
class RenderingOptions:
    # Language of the translated text; chooses between font_path and cjk_font_path.
    target_language: Language = Language.EN
    # Font for Latin, Cyrillic and Vietnamese text, downloaded by `download-models`.
    font_path: Path = Path("fonts/NotoSans-Regular.ttf")
    # Font for Chinese, Korean and Japanese text, downloaded by `download-models`.
    cjk_font_path: Path = Path("fonts/NotoSansCJKjp-Regular.otf")
    # Smallest font size; text that does not fit even at this size overflows its region.
    min_font_size: int = 8
    max_font_size: int = 48

    def __post_init__(self) -> None:
        if not 0 < self.min_font_size <= self.max_font_size:
            raise ValueError("font sizes must satisfy 0 < min_font_size <= max_font_size")


@dataclass(frozen=True)
class OutputOptions:
    # Directory receiving the translated pages, named like their source files.
    output_dir: Path = Path("output")


@dataclass(frozen=True)
class AppConfig:
    input_path: Path
    image: ImageOptions = field(default_factory=ImageOptions)
    detection: DetectionOptions = field(default_factory=DetectionOptions)
    recognition: RecognitionOptions = field(default_factory=RecognitionOptions)
    translation: TranslationOptions = field(default_factory=TranslationOptions)
    erasing: ErasingOptions = field(default_factory=ErasingOptions)
    rendering: RenderingOptions = field(default_factory=RenderingOptions)
    output: OutputOptions = field(default_factory=OutputOptions)
