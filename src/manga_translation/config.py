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
class AppConfig:
    input_path: Path
    image: ImageOptions = field(default_factory=ImageOptions)
    detection: DetectionOptions = field(default_factory=DetectionOptions)
    recognition: RecognitionOptions = field(default_factory=RecognitionOptions)
    translation: TranslationOptions = field(default_factory=TranslationOptions)
