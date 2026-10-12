"""Downloading the model weights and fonts to the paths the configuration expects."""

import shutil
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .config.options import AppConfig, Language

_HUGGING_FACE = "https://huggingface.co/{repo}/resolve/main/{file}"
_DETECTOR_URL = (
    "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx"
)
_FONT_URL = "https://github.com/notofonts/notofonts.github.io/raw/main/fonts/NotoSans/hinted/ttf/NotoSans-Regular.ttf"
_CJK_FONT_URL = "https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/Japanese/NotoSansCJKjp-Regular.otf"
# Directory of the PaddleOCR export in monkt/paddleocr-onnx for each language it reads.
_PADDLE_LANGUAGES = {Language.EN: "english", Language.KO: "korean", Language.ZH: "chinese"}


class DownloadError(OSError):
    """Raised when a file cannot be downloaded."""


@dataclass(frozen=True)
class Download:
    url: str
    path: Path


def _hugging_face(repo: str, file: str, path: Path) -> Download:
    return Download(_HUGGING_FACE.format(repo=repo, file=file), path)


def required_downloads(config: AppConfig) -> list[Download]:
    """Every model and font file the pipeline needs, saved where the config looks for it."""
    ocr = config.recognition.manga_ocr_dir
    nllb = config.translation.model_dir
    downloads = [
        Download(_DETECTOR_URL, config.detection.model_path),
        _hugging_face("dnouv/manga-ocr", "onnx/encoder_model.onnx", ocr / "encoder_model.onnx"),
        _hugging_face("dnouv/manga-ocr", "onnx/decoder_model_merged.onnx", ocr / "decoder_model_merged.onnx"),
        _hugging_face("dnouv/manga-ocr", "vocab.txt", ocr / "vocab.txt"),
    ]
    for language, name in _PADDLE_LANGUAGES.items():
        directory = config.recognition.paddle_ocr_dir / language
        downloads.append(_hugging_face("monkt/paddleocr-onnx", f"languages/{name}/rec.onnx", directory / "model.onnx"))
        downloads.append(_hugging_face("monkt/paddleocr-onnx", f"languages/{name}/dict.txt", directory / "dict.txt"))
    downloads += [
        _hugging_face("Xenova/nllb-200-distilled-600M", "onnx/encoder_model.onnx", nllb / "encoder_model.onnx"),
        _hugging_face("Xenova/nllb-200-distilled-600M", "onnx/decoder_model.onnx", nllb / "decoder_model.onnx"),
        _hugging_face("Xenova/nllb-200-distilled-600M", "tokenizer.json", nllb / "tokenizer.json"),
        _hugging_face("Carve/LaMa-ONNX", "lama_fp32.onnx", config.erasing.model_path),
        Download(_FONT_URL, config.rendering.font_path),
        Download(_CJK_FONT_URL, config.rendering.cjk_font_path),
    ]
    return downloads


def _fetch(url: str, path: Path) -> None:
    """Write the resource at the URL to the file."""
    with urllib.request.urlopen(url) as response, path.open("wb") as file:
        shutil.copyfileobj(response, file)


def download(item: Download, fetch: Callable[[str, Path], None] = _fetch) -> bool:
    """Download the file unless it exists; returns whether it was downloaded.

    The file is written under a temporary name first, so an interrupted download never leaves
    a partial file under the final name.
    """
    if item.path.exists():
        return False
    item.path.parent.mkdir(parents=True, exist_ok=True)
    partial = item.path.with_name(item.path.name + ".part")
    try:
        fetch(item.url, partial)
    except OSError as error:
        partial.unlink(missing_ok=True)
        raise DownloadError(f"{item.url}: {error}") from error
    partial.replace(item.path)
    return True
