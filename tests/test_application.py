import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from manga_translation import AppConfig, ImageOptions
from manga_translation.application import run
from manga_translation.steps.image_loading import get_image
from manga_translation.steps.text_detection import DetectionResult


class RunTests(unittest.TestCase):
    def setUp(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.directory = Path(temp_dir.name)
        detect = patch(
            "manga_translation.steps.text_detection.ComicTextDetector.detect",
            side_effect=lambda image: DetectionResult([], Image.new("L", image.size)),
        )
        detect.start()
        self.addCleanup(detect.stop)

    def save_image(self, name: str, size: tuple[int, int] = (30, 20)) -> Path:
        path = self.directory / name
        Image.new("RGB", size).save(path)
        return path

    def test_returns_one_context_per_image_in_order(self) -> None:
        self.save_image("b.png")
        self.save_image("a.png")
        contexts = run(AppConfig(self.directory))
        self.assertEqual([context.source_path.name for context in contexts], ["a.png", "b.png"])

    def test_image_options_reach_steps(self) -> None:
        path = self.save_image("page.png", (300, 100))
        [context] = run(AppConfig(path, ImageOptions(max_size=150)))
        self.assertEqual(get_image(context).size, (150, 50))
