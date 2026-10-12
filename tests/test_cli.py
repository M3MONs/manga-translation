import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from typer.testing import CliRunner

from manga_translation import (
    AppConfig,
    ImageOptions,
    Language,
    OutputOptions,
    RenderingOptions,
    TranslationOptions,
)
from manga_translation.cli import app, run
from manga_translation.downloads import DownloadError
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
        return CliRunner().invoke(app, ["translate", *args, "--output", str(self.output.output_dir)])

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
                path,
                translation=TranslationOptions(target_language=Language.PL),
                rendering=RenderingOptions(target_language=Language.PL),
                output=self.output,
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
            result = CliRunner().invoke(app, ["translate", str(path)])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(fake_run.call_args.args[0].output, OutputOptions(Path("output")))

    def test_output_option_writes_page_there(self) -> None:
        path = self.save_image("page.png")
        result = self.invoke(str(path))
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue((self.output.output_dir / "page.png").is_file())


class CliConfigFileTests(unittest.TestCase):
    def setUp(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.directory = Path(temp_dir.name)
        self.page = self.directory / "page.png"
        Image.new("RGB", (30, 20)).save(self.page)
        for target in ("manga_translation.cli.configure_logging", "manga_translation.cli.run"):
            patcher = patch(target, return_value=[])
            self.addCleanup(patcher.stop)
            self.run_mock = patcher.start()

    def write_config(self, text: str) -> Path:
        path = self.directory / "settings.yaml"
        path.write_text(text, "utf-8")
        return path

    def invoke(self, *args: str):
        return CliRunner().invoke(app, ["translate", str(self.page), *args])

    def config_used(self) -> AppConfig:
        return self.run_mock.call_args.args[0]

    def test_config_file_settings_reach_config(self) -> None:
        path = self.write_config("image:\n  max_size: 200\nrendering:\n  max_font_size: 30\n")
        result = self.invoke("--config", str(path))
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(self.config_used().image.max_size, 200)
        self.assertEqual(self.config_used().rendering.max_font_size, 30)

    def test_options_override_config_file(self) -> None:
        path = self.write_config(
            "target_language: de\nimage:\n  max_size: 200\noutput:\n  output_dir: from_file\n"
        )
        result = self.invoke(
            "--config", str(path), "--max-size", "100", "--language", "pl", "--output", "from_cli"
        )
        self.assertEqual(result.exit_code, 0, result.output)
        config = self.config_used()
        self.assertEqual(config.image.max_size, 100)
        self.assertEqual(config.translation.target_language, Language.PL)
        self.assertEqual(config.rendering.target_language, Language.PL)
        self.assertEqual(config.output.output_dir, Path("from_cli"))

    def test_source_language_option_overrides_config_file(self) -> None:
        path = self.write_config("source_language: zh\n")
        result = self.invoke("--config", str(path), "--source-language", "ko")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(self.config_used().recognition.source_language, Language.KO)
        self.assertEqual(self.config_used().translation.source_language, Language.KO)

    def test_source_language_from_config_file_is_kept(self) -> None:
        result = self.invoke("--config", str(self.write_config("source_language: en\n")))
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(self.config_used().recognition.source_language, Language.EN)

    def test_unsupported_source_language_option_fails(self) -> None:
        result = self.invoke("--source-language", "pl")
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("source_language", result.output)
        self.run_mock.assert_not_called()

    def test_japanese_target_language_is_accepted(self) -> None:
        result = self.invoke("--language", "ja")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(self.config_used().translation.target_language, Language.JA)
        self.assertEqual(self.config_used().rendering.target_language, Language.JA)

    def test_default_config_file_is_used_when_present(self) -> None:
        path = self.write_config("image:\n  max_size: 300\n")
        with patch("manga_translation.cli.DEFAULT_CONFIG_PATH", path):
            result = self.invoke()
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(self.config_used().image.max_size, 300)

    def test_defaults_are_used_without_config_file(self) -> None:
        with patch("manga_translation.cli.DEFAULT_CONFIG_PATH", self.directory / "missing.yaml"):
            result = self.invoke()
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(self.config_used(), AppConfig(self.page))

    def test_missing_config_file_fails(self) -> None:
        result = self.invoke("--config", str(self.directory / "missing.yaml"))
        self.assertNotEqual(result.exit_code, 0)
        self.run_mock.assert_not_called()

    def test_invalid_config_file_fails(self) -> None:
        invalid = {
            "wrong type": "image:\n  max_size: abc\n",
            "value out of range": "image:\n  max_size: -1\n",
        }
        for name, text in invalid.items():
            with self.subTest(name):
                result = self.invoke("--config", str(self.write_config(text)))
                self.assertNotEqual(result.exit_code, 0)
                self.assertIn("max_size", result.output)
        self.run_mock.assert_not_called()


class CliDownloadModelsTests(unittest.TestCase):
    def setUp(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.directory = Path(temp_dir.name)
        patcher = patch("manga_translation.cli.DEFAULT_CONFIG_PATH", self.directory / "missing.yaml")
        patcher.start()
        self.addCleanup(patcher.stop)

    def invoke(self, *args: str):
        return CliRunner().invoke(app, ["download-models", *args])

    def test_downloads_to_config_paths(self) -> None:
        config = self.directory / "settings.yaml"
        config.write_text("detection:\n  model_path: custom/detector.onnx\n", "utf-8")
        with patch("manga_translation.cli.download", return_value=True) as fake_download:
            result = self.invoke("--config", str(config))
        self.assertEqual(result.exit_code, 0, result.output)
        paths = [call.args[0].path for call in fake_download.call_args_list]
        self.assertIn(Path("custom/detector.onnx"), paths)
        self.assertIn(f"{Path('custom/detector.onnx')}: downloaded\n", result.output)

    def test_existing_files_are_reported(self) -> None:
        with patch("manga_translation.cli.download", return_value=False):
            result = self.invoke()
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn(f"{Path('models/comictextdetector.pt.onnx')}: exists\n", result.output)

    def test_failed_download_stops(self) -> None:
        error = DownloadError("https://example.com/model.onnx: connection reset")
        with patch("manga_translation.cli.download", side_effect=error) as fake_download:
            result = self.invoke()
        self.assertEqual(result.exit_code, 1)
        self.assertIn("connection reset", result.output)
        fake_download.assert_called_once()
