import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from typer.testing import CliRunner

from manga_translation import AppConfig, ImageOptions, Language, OutputOptions, TranslationOptions
from manga_translation.cli import app, run
from manga_translation.steps.text_detection import DetectionResult


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.directory = Path(temp_dir.name)
        self.output = OutputOptions(self.directory / "out")
        patcher = patch("manga_translation.cli.configure_logging")
        patcher.start()
        self.addCleanup(patcher.stop)
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

    def invoke(self, *args: str):
        return CliRunner().invoke(app, [*args, "--output", str(self.output.output_dir)])

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
        self.assertEqual(wrapped_run.call_args.args[0], AppConfig(path, ImageOptions(max_size=150), output=self.output))

    def test_language_defaults_to_english(self) -> None:
        path = self.save_image("page.png")
        with patch("manga_translation.cli.run", wraps=run) as wrapped_run:
            result = self.invoke(str(path))
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(wrapped_run.call_args.args[0].translation.target_language, Language.EN)

    def test_language_option_reaches_config(self) -> None:
        path = self.save_image("page.png")
        with patch("manga_translation.cli.run", wraps=run) as wrapped_run:
            result = self.invoke(str(path), "--language", "pl")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            wrapped_run.call_args.args[0],
            AppConfig(
                path, translation=TranslationOptions(target_language=Language.PL), output=self.output
            ),
        )

    def test_unsupported_language_fails(self) -> None:
        path = self.save_image("page.png")
        result = self.invoke(str(path), "--language", "xx")
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("xx", result.output)

    def test_output_defaults_to_output_directory(self) -> None:
        path = self.save_image("page.png")
        with patch("manga_translation.cli.run") as fake_run:
            result = CliRunner().invoke(app, [str(path)])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(fake_run.call_args.args[0].output, OutputOptions(Path("output")))

    def test_output_option_writes_page_there(self) -> None:
        path = self.save_image("page.png")
        result = self.invoke(str(path))
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue((self.output.output_dir / "page.png").is_file())
