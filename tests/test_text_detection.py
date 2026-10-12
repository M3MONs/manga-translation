import tempfile
import unittest
from pathlib import Path
from typing import cast

import numpy as np
import onnxruntime
from PIL import Image

from manga_translation import DetectionOptions, PageContext, Pipeline
from manga_translation.core.regions import BoundingBox, TextRegion
from manga_translation.steps.text_detection import ComicTextDetector, DetectText, TextDetection


class FakeSession:
    """Returns prepared model outputs; boxes are rows of [cx, cy, w, h, objectness, cls0, cls1]."""

    def __init__(self, blocks: list[list[float]], seg_value: float = 0.0) -> None:
        self.blocks = np.array(blocks, dtype=np.float32).reshape(1, -1, 7)
        self.seg = np.full((1, 1, 1024, 1024), seg_value, dtype=np.float32)
        self.inputs: np.ndarray | None = None

    def run(self, output_names, feeds):
        self.inputs = feeds["images"]
        return [self.blocks, self.seg]


def use_session(detector: ComicTextDetector, session: FakeSession) -> ComicTextDetector:
    detector.session = cast(onnxruntime.InferenceSession, session)
    return detector


def detector(blocks: list[list[float]], seg_value: float = 0.0, **options) -> ComicTextDetector:
    return use_session(ComicTextDetector(DetectionOptions(**options)), FakeSession(blocks, seg_value))


class ComicTextDetectorTests(unittest.TestCase):
    def test_low_confidence_is_discarded(self) -> None:
        regions = detector(
            [[100, 100, 20, 20, 0.9, 1, 0], [300, 300, 20, 20, 0.9, 0.3, 0.2]]
        ).detect(Image.new("RGB", (1024, 1024))).regions
        self.assertEqual(len(regions), 1)
        self.assertAlmostEqual(regions[0].confidence, 0.9)

    def test_overlapping_boxes_keep_the_best(self) -> None:
        regions = detector(
            [[100, 100, 40, 40, 0.8, 1, 0], [102, 102, 40, 40, 0.95, 1, 0], [500, 500, 40, 40, 0.7, 0, 1]]
        ).detect(Image.new("RGB", (1024, 1024))).regions
        self.assertEqual([round(region.confidence, 2) for region in regions], [0.95, 0.7])

    def test_boxes_are_scaled_to_page_and_clipped(self) -> None:
        # A 2048x1024 page is scaled by 0.5 into the top half of the model input.
        regions = detector(
            [[100, 50, 20, 10, 1, 1, 0], [1020, 500, 20, 20, 0.5, 1, 0]]
        ).detect(Image.new("RGB", (2048, 1024))).regions
        self.assertEqual(
            regions,
            [
                TextRegion(BoundingBox(180, 90, 220, 110), 1.0),
                TextRegion(BoundingBox(2020, 980, 2048, 1020), 0.5),
            ],
        )

    def test_input_is_letterboxed_rgb_tensor(self) -> None:
        session = FakeSession([])
        use_session(ComicTextDetector(DetectionOptions()), session).detect(
            Image.new("RGB", (512, 256), (255, 0, 0))
        )
        tensor = session.inputs
        assert tensor is not None
        self.assertEqual(tensor.shape, (1, 3, 1024, 1024))
        self.assertEqual(tensor[0, :, 0, 0].tolist(), [1, 0, 0])
        self.assertEqual(tensor[0, :, 600, 0].tolist(), [0, 0, 0])

    def test_mask_matches_page_size(self) -> None:
        mask = detector([], seg_value=1.0).detect(Image.new("RGB", (300, 200))).mask
        self.assertEqual((mask.mode, mask.size), ("L", (300, 200)))
        self.assertEqual(mask.getpixel((299, 199)), 255)

    def test_missing_weights_are_reported(self) -> None:
        text_detector = ComicTextDetector(DetectionOptions(model_path=Path("missing.onnx")))
        with self.assertRaisesRegex(FileNotFoundError, "comictextdetector.pt.onnx"):
            text_detector.detect(Image.new("RGB", (10, 10)))


class TextDetectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_step_stores_regions_and_mask(self) -> None:
        group = TextDetection(DetectionOptions())
        [step] = group.steps()
        assert isinstance(step, DetectText)
        use_session(step.detector, FakeSession([[100, 100, 20, 20, 1, 1, 0]]))
        context = PageContext(Path("page.png"), {"image": Image.new("RGB", (1024, 1024))})
        await Pipeline([step]).run(context)
        self.assertEqual(
            context.artifacts["text_regions"], [TextRegion(BoundingBox(90, 90, 110, 110), 1.0)]
        )
        self.assertIsInstance(context.artifacts["text_mask"], Image.Image)

    async def test_step_logs_detected_regions(self) -> None:
        [step] = TextDetection(DetectionOptions()).steps()
        assert isinstance(step, DetectText)
        use_session(step.detector, FakeSession([[100, 100, 20, 20, 1, 1, 0]]))
        context = PageContext(Path("page.png"), {"image": Image.new("RGB", (1024, 1024))})
        with self.assertLogs("manga_translation.steps.text_detection", "DEBUG") as logs:
            await step.execute(context)
        self.assertEqual(
            logs.output,
            [
                "INFO:manga_translation.steps.text_detection:Text detected page=page.png regions=1",
                "DEBUG:manga_translation.steps.text_detection:"
                "Text region page=page.png box=(90, 90, 110, 110) confidence=1.00",
            ],
        )
