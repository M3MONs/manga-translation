import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from typer.testing import CliRunner

from manga_translation import AppConfig, ImageOptions
from manga_translation.cli import app, run


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.directory = Path(temp_dir.name)
        patcher = patch("manga_translation.cli.configure_logging")
        patcher.start()
        self.addCleanup(patcher.stop)

    def save_image(self, name: str, size: tuple[int, int] = (30, 20)) -> Path:
        path = self.directory / name
        Image.new("RGB", size).save(path)
        return path

    def invoke(self, *args: str):
        return CliRunner().invoke(app, [*args])

    def test_single_file(self) -> None:
        path = self.save_image("page.png")
        result = self.invoke(str(path))
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(result.output, f"{path}: done\n")

    def test_directory(self) -> None:
        first = self.save_image("a.png")
        second = self.save_image("b.png")
        result = self.invoke(str(self.directory))
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(result.output, f"{first}: done\n{second}: done\n")

    def test_unsupported_file_fails(self) -> None:
        path = self.directory / "page.bmp"
        Image.new("RGB", (5, 5)).save(path)
        result = self.invoke(str(path))
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("unsupported image format", result.output)

    def test_empty_directory_fails(self) -> None:
        result = self.invoke(str(self.directory))
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("no supported images", result.output)

    def test_max_size_option_reaches_config(self) -> None:
        path = self.save_image("page.png", (300, 100))
        with patch("manga_translation.cli.run", wraps=run) as wrapped_run:
            result = self.invoke(str(path), "--max-size", "150")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(wrapped_run.call_args.args[0], AppConfig(path, ImageOptions(max_size=150)))
