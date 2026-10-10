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
    ID = "id"
    VI = "vi"
    TR = "tr"


@dataclass(frozen=True)
class ImageOptions:
    max_size: int = 4096
    """Longest allowed image side in pixels; 0 disables resizing."""

    def __post_init__(self) -> None:
        if self.max_size < 0:
            raise ValueError("max_size must be >= 0")


@dataclass(frozen=True)
class DetectionOptions:
    model_path: Path = Path("models/comictextdetector.pt.onnx")
    """comic-text-detector ONNX weights, downloaded manually."""
    confidence_threshold: float = 0.4
    """Detections scoring below this are discarded."""
    nms_threshold: float = 0.35
    """IoU above which overlapping detections are merged into the best one."""

    def __post_init__(self) -> None:
        for name in ("confidence_threshold", "nms_threshold"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True)
class RecognitionOptions:
    model_dir: Path = Path("models/manga-ocr")
    """manga-ocr ONNX export (encoder, merged decoder, vocab.txt), downloaded manually."""
    max_length: int = 300
    """Most tokens decoded per text region, including the start token."""

    def __post_init__(self) -> None:
        if self.max_length <= 0:
            raise ValueError("max_length must be > 0")


@dataclass(frozen=True)
class TranslationOptions:
    target_language: Language = Language.EN
    model_dir: Path = Path("models/nllb-200-distilled-600M")
    """NLLB ONNX export (encoder, decoder, tokenizer.json), downloaded manually."""
    max_length: int = 256
    """Most tokens decoded per text region, including the forced prefix."""

    def __post_init__(self) -> None:
        if self.max_length <= 0:
            raise ValueError("max_length must be > 0")


@dataclass(frozen=True)
class ErasingOptions:
    model_path: Path = Path("models/lama/lama_fp32.onnx")
    """LaMa inpainting ONNX weights, downloaded manually."""
    mask_threshold: int = 64
    """Text mask values above this (0–255) count as text pixels."""
    mask_dilation: int = 21
    """Odd size of the filter growing the text mask to cover glyph outlines; 1 disables it."""

    def __post_init__(self) -> None:
        if not 0 <= self.mask_threshold <= 255:
            raise ValueError("mask_threshold must be between 0 and 255")
        if self.mask_dilation < 1 or self.mask_dilation % 2 == 0:
            raise ValueError("mask_dilation must be an odd number >= 1")


@dataclass(frozen=True)
class RenderingOptions:
    target_language: Language = Language.EN
    """Language of the translated text; chooses between font_path and cjk_font_path."""
    font_path: Path = Path("models/fonts/NotoSans-Regular.ttf")
    """Font for Latin, Cyrillic and Vietnamese text, downloaded manually."""
    cjk_font_path: Path = Path("models/fonts/NotoSansCJKjp-Regular.otf")
    """Font for Chinese and Korean text, downloaded manually."""
    min_font_size: int = 8
    """Smallest font size; text that does not fit even at this size overflows its region."""
    max_font_size: int = 48

    def __post_init__(self) -> None:
        if not 0 < self.min_font_size <= self.max_font_size:
            raise ValueError("font sizes must satisfy 0 < min_font_size <= max_font_size")


@dataclass(frozen=True)
class OutputOptions:
    output_dir: Path = Path("output")
    """Directory receiving the translated pages, named like their source files."""


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
