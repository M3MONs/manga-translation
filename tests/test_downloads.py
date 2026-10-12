import tempfile
import unittest
from pathlib import Path

from manga_translation import AppConfig, DetectionOptions, RecognitionOptions, RenderingOptions
from manga_translation.downloads import Download, DownloadError, download, required_downloads


class RequiredDownloadsTests(unittest.TestCase):
    def test_paths_follow_config(self) -> None:
        config = AppConfig(
            Path("."),
            detection=DetectionOptions(model_path=Path("custom/detector.onnx")),
            recognition=RecognitionOptions(paddle_ocr_dir=Path("custom/paddle")),
            rendering=RenderingOptions(font_path=Path("custom/font.ttf")),
        )

        paths = {item.path for item in required_downloads(config)}

        self.assertIn(Path("custom/detector.onnx"), paths)
        self.assertIn(Path("custom/paddle/ko/model.onnx"), paths)
        self.assertIn(Path("custom/paddle/zh/dict.txt"), paths)
        self.assertIn(Path("custom/font.ttf"), paths)

    def test_every_model_file_is_listed_once(self) -> None:
        downloads = required_downloads(AppConfig(Path(".")))

        paths = [item.path for item in downloads]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertEqual(len(paths), 16)
        self.assertTrue(all(item.url.startswith("https://") for item in downloads))


class DownloadTests(unittest.TestCase):
    def setUp(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.directory = Path(temp_dir.name)
        self.item = Download("https://example.com/model.onnx", self.directory / "models" / "model.onnx")

    def test_downloads_into_new_directory(self) -> None:
        fetched = []

        def fetch(url: str, path: Path) -> None:
            fetched.append(url)
            path.write_bytes(b"weights")

        self.assertTrue(download(self.item, fetch))
        self.assertEqual(fetched, [self.item.url])
        self.assertEqual(self.item.path.read_bytes(), b"weights")
        self.assertEqual([path.name for path in self.item.path.parent.iterdir()], ["model.onnx"])

    def test_existing_file_is_skipped(self) -> None:
        self.item.path.parent.mkdir()
        self.item.path.write_bytes(b"old")

        def fetch(url: str, path: Path) -> None:
            self.fail("existing file was downloaded again")

        self.assertFalse(download(self.item, fetch))
        self.assertEqual(self.item.path.read_bytes(), b"old")

    def test_failure_leaves_no_file(self) -> None:
        def fetch(url: str, path: Path) -> None:
            path.write_bytes(b"half")
            raise OSError("connection reset")

        with self.assertRaisesRegex(DownloadError, "example.com.*connection reset"):
            download(self.item, fetch)
        self.assertEqual(list(self.item.path.parent.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
