import unittest
from pathlib import Path

from manga_translation import AppConfig, ImageOptions


class ConfigTests(unittest.TestCase):
    def test_negative_max_size_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ImageOptions(max_size=-1)

    def test_defaults(self) -> None:
        config = AppConfig(Path("page.png"))
        self.assertEqual(config.image, ImageOptions(max_size=4096))
