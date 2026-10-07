import unittest
from pathlib import Path

from manga_translation import AppConfig, DetectionOptions, ImageOptions


class ConfigTests(unittest.TestCase):
    def test_negative_max_size_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ImageOptions(max_size=-1)

    def test_defaults(self) -> None:
        config = AppConfig(Path("page.png"))
        self.assertEqual(config.image, ImageOptions(max_size=4096))
        self.assertEqual(config.detection, DetectionOptions())

    def test_detection_thresholds_outside_unit_range_are_rejected(self) -> None:
        invalid = {
            "confidence below 0": lambda: DetectionOptions(confidence_threshold=-0.1),
            "confidence above 1": lambda: DetectionOptions(confidence_threshold=1.1),
            "nms above 1": lambda: DetectionOptions(nms_threshold=2),
        }
        for name, create in invalid.items():
            with self.subTest(name), self.assertRaises(ValueError):
                create()
