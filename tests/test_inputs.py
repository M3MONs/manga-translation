import tempfile
import unittest
from pathlib import Path

from PIL import Image

from manga_translation.inputs import InputError, collect_image_paths


class CollectImagePathsTests(unittest.TestCase):
    def setUp(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.directory = Path(temp_dir.name)

    def save_image(self, name: str) -> Path:
        path = self.directory / name
        Image.new("RGB", (5, 5)).save(path)
        return path

    def test_single_file(self) -> None:
        path = self.save_image("page.png")
        self.assertEqual(collect_image_paths(path), [path])

    def test_directory_is_flat_and_sorted(self) -> None:
        self.save_image("b.JPG")
        self.save_image("a.png")
        (self.directory / "notes.txt").write_text("skip")
        (self.directory / "nested").mkdir()
        Image.new("RGB", (5, 5)).save(self.directory / "nested" / "c.png")
        paths = collect_image_paths(self.directory)
        self.assertEqual([path.name for path in paths], ["a.png", "b.JPG"])

    def test_unsupported_file_fails(self) -> None:
        path = self.save_image("page.bmp")
        with self.assertRaisesRegex(InputError, "unsupported image format"):
            collect_image_paths(path)

    def test_empty_directory_fails(self) -> None:
        with self.assertRaisesRegex(InputError, "no supported images"):
            collect_image_paths(self.directory)
