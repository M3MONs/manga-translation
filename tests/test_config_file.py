import unittest
from pathlib import Path

from manga_translation import AppConfig, DetectionOptions, Language
from manga_translation.config_file import ConfigError, build_config

PAGE = Path("page.png")


class BuildConfigTests(unittest.TestCase):
    def test_empty_data_gives_defaults(self) -> None:
        self.assertEqual(build_config(PAGE, {}), AppConfig(PAGE))

    def test_empty_section_gives_defaults(self) -> None:
        self.assertEqual(build_config(PAGE, {"image": None}), AppConfig(PAGE))

    def test_settings_are_converted_to_field_types(self) -> None:
        config = build_config(
            PAGE,
            {"detection": {"model_path": "x.onnx", "confidence_threshold": 1, "nms_threshold": 0.5}},
        )
        self.assertEqual(
            config.detection,
            DetectionOptions(Path("x.onnx"), confidence_threshold=1.0, nms_threshold=0.5),
        )
        self.assertIsInstance(config.detection.confidence_threshold, float)

    def test_unset_settings_keep_defaults(self) -> None:
        config = build_config(PAGE, {"rendering": {"max_font_size": 30}})
        self.assertEqual(config.rendering.max_font_size, 30)
        self.assertEqual(config.rendering.min_font_size, 8)

    def test_target_language_reaches_translation_and_rendering(self) -> None:
        config = build_config(PAGE, {"target_language": "pl"})
        self.assertEqual(config.translation.target_language, Language.PL)
        self.assertEqual(config.rendering.target_language, Language.PL)

    def test_invalid_data_is_rejected(self) -> None:
        invalid = {
            "unknown section": {"colors": {}},
            "unknown key": {"image": {"width": 10}},
            "section not a mapping": {"image": 10},
            "text for int": {"image": {"max_size": "abc"}},
            "bool for int": {"image": {"max_size": True}},
            "float for int": {"image": {"max_size": 1.5}},
            "number for path": {"output": {"output_dir": 5}},
            "unsupported language": {"target_language": "xx"},
            "language inside section": {"translation": {"target_language": "pl"}},
        }
        for name, data in invalid.items():
            with self.subTest(name), self.assertRaises(ConfigError):
                build_config(PAGE, data)

    def test_error_names_the_setting(self) -> None:
        with self.assertRaisesRegex(ConfigError, r"image\.max_size"):
            build_config(PAGE, {"image": {"max_size": "abc"}})

    def test_option_validation_still_applies(self) -> None:
        with self.assertRaises(ValueError):
            build_config(PAGE, {"image": {"max_size": -1}})
