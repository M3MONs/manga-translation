"""Steps that find text on a page with comic-text-detector."""

import logging
from dataclasses import dataclass
from functools import cached_property
from typing import cast

import numpy as np
import onnxruntime
from PIL import Image

from ..config import DetectionOptions
from ..context import PageContext
from ..regions import BoundingBox, TextRegion
from ..step import Step, StepGroup
from .image_loading import get_image

_INPUT_SIZE = 1024
_WEIGHTS_URL = "https://github.com/zyddnys/manga-image-translator/releases/tag/beta-0.3"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectionResult:
    regions: list[TextRegion]
    mask: Image.Image
    """Per-pixel text probability scaled to 0-255, mode L, same size as the page."""


class ComicTextDetector:
    """Runs comic-text-detector ONNX weights on a whole page; the model is loaded on first use."""

    def __init__(self, options: DetectionOptions) -> None:
        self.options = options

    @cached_property
    def session(self) -> onnxruntime.InferenceSession:
        path = self.options.model_path
        if not path.is_file():
            raise FileNotFoundError(
                f"Text detection weights not found at {path}; "
                f"download comictextdetector.pt.onnx from {_WEIGHTS_URL}"
            )
        return onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])

    def detect(self, image: Image.Image) -> DetectionResult:
        tensor, scale, (width, height) = _to_model_input(image)

        # The model returns dense arrays; the stubs also allow sparse tensors, lists and dicts.
        blocks, seg = cast(list[np.ndarray], self.session.run(["blk", "seg"], {"images": tensor}))

        mask = (seg[0, 0, :height, :width] * 255).clip(0, 255).astype(np.uint8)
        return DetectionResult(
            regions=self._regions(blocks[0], scale, image.size),
            mask=Image.fromarray(mask, "L").resize(image.size, Image.Resampling.BILINEAR),
        )

    def _regions(
        self, blocks: np.ndarray, scale: float, page_size: tuple[int, int]
    ) -> list[TextRegion]:
        """Turn raw [cx, cy, w, h, objectness, class scores...] rows into page regions."""
        scores = blocks[:, 4] * blocks[:, 5:].max(axis=1)
        keep = scores >= self.options.confidence_threshold
        blocks, scores = blocks[keep], scores[keep]
        centers, sizes = blocks[:, :2], blocks[:, 2:4]
        boxes = np.concatenate([centers - sizes / 2, centers + sizes / 2], axis=1) / scale

        page_width, page_height = page_size
        regions = []
        for index in _non_max_suppression(boxes, scores, self.options.nms_threshold):
            x1, y1, x2, y2 = boxes[index]
            box = BoundingBox(
                x1=int(np.clip(np.floor(x1), 0, page_width)),
                y1=int(np.clip(np.floor(y1), 0, page_height)),
                x2=int(np.clip(np.ceil(x2), 0, page_width)),
                y2=int(np.clip(np.ceil(y2), 0, page_height)),
            )
            if box.x2 > box.x1 and box.y2 > box.y1:
                regions.append(TextRegion(box, float(scores[index])))
        return regions


def _to_model_input(image: Image.Image) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Letterbox the page into the model's 1024x1024 NCHW float input; the page itself is untouched.

    Returns the tensor, the page-to-input scale and the scaled page size inside the input.
    """
    scale = _INPUT_SIZE / max(image.size)
    width, height = (max(1, round(side * scale)) for side in image.size)
    canvas = Image.new("RGB", (_INPUT_SIZE, _INPUT_SIZE))
    canvas.paste(image.resize((width, height), Image.Resampling.BILINEAR))
    tensor = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[np.newaxis] / 255
    return tensor, scale, (width, height)


def _non_max_suppression(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Return indices of the best boxes, dropping any overlapping a better one above the threshold."""
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = scores.argsort()[::-1]
    kept = []
    while order.size:
        best, rest = order[0], order[1:]
        kept.append(int(best))
        top_left = np.maximum(boxes[best, :2], boxes[rest, :2])
        bottom_right = np.minimum(boxes[best, 2:], boxes[rest, 2:])
        intersection = (bottom_right - top_left).clip(min=0).prod(axis=1)
        iou = intersection / (areas[best] + areas[rest] - intersection)
        order = rest[iou <= iou_threshold]
    return kept


class DetectText:
    """Requires image; produces text_regions (list[TextRegion]) and text_mask (L image)."""

    def __init__(self, detector: ComicTextDetector) -> None:
        self.detector = detector

    async def execute(self, context: PageContext) -> None:
        result = self.detector.detect(get_image(context))
        context.artifacts["text_regions"] = result.regions
        context.artifacts["text_mask"] = result.mask
        logger.info("Text detected page=%s regions=%d", context.source_path, len(result.regions))
        for region in result.regions:
            box = region.box
            logger.info(
                "Text region page=%s box=(%d, %d, %d, %d) confidence=%.2f",
                context.source_path, box.x1, box.y1, box.x2, box.y2, region.confidence,
            )


class TextDetection(StepGroup):
    """Detect text blocks and the text mask on the whole page."""

    def __init__(self, options: DetectionOptions) -> None:
        self.options = options

    def steps(self) -> list[Step]:
        return [DetectText(ComicTextDetector(self.options))]
