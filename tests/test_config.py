import unittest
from pathlib import Path

from manga_translation import (
    AppConfig,
    DetectionOptions,
    ErasingOptions,
    ImageOptions,
    Language,
    OutputOptions,
    RecognitionOptions,
    RenderingOptions,
    TranslationOptions,
)


class ConfigTests(unittest.TestCase):
    def test_negative_max_size_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ImageOptions(max_size=-1)

    def test_defaults(self) -> None:
        config = AppConfig(Path("page.png"))
        self.assertEqual(config.image, ImageOptions(max_size=4096))
        self.assertEqual(config.detection, DetectionOptions())
        self.assertEqual(config.recognition, RecognitionOptions())
        self.assertEqual(config.translation, TranslationOptions())
        self.assertEqual(config.translation.target_language, Language.EN)
        self.assertEqual(config.erasing, ErasingOptions(mask_threshold=64, mask_dilation=21))
        self.assertEqual(config.rendering, RenderingOptions(Language.EN, min_font_size=8, max_font_size=48))
        self.assertEqual(config.output, OutputOptions(Path("output")))

    def test_detection_thresholds_outside_unit_range_are_rejected(self) -> None:
        invalid = {
            "confidence below 0": lambda: DetectionOptions(confidence_threshold=-0.1),
            "confidence above 1": lambda: DetectionOptions(confidence_threshold=1.1),
            "nms above 1": lambda: DetectionOptions(nms_threshold=2),
        }
        for name, create in invalid.items():
            with self.subTest(name), self.assertRaises(ValueError):
                create()

    def test_non_positive_max_length_is_rejected(self) -> None:
        for max_length in (0, -1):
            with self.subTest(max_length), self.assertRaises(ValueError):
                RecognitionOptions(max_length=max_length)
            with self.subTest(max_length), self.assertRaises(ValueError):
                TranslationOptions(max_length=max_length)

    def test_invalid_erasing_options_are_rejected(self) -> None:
        invalid = {
            "threshold below 0": lambda: ErasingOptions(mask_threshold=-1),
            "threshold above 255": lambda: ErasingOptions(mask_threshold=256),
            "dilation below 1": lambda: ErasingOptions(mask_dilation=0),
            "even dilation": lambda: ErasingOptions(mask_dilation=4),
        }
        for name, create in invalid.items():
            with self.subTest(name), self.assertRaises(ValueError):
                create()

    def test_invalid_font_sizes_are_rejected(self) -> None:
        invalid = {
            "min not positive": lambda: RenderingOptions(min_font_size=0),
            "min above max": lambda: RenderingOptions(min_font_size=20, max_font_size=10),
        }
        for name, create in invalid.items():
            with self.subTest(name), self.assertRaises(ValueError):
                create()
