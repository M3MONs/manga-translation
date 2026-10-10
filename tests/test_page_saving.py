import hashlib
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from manga_translation import OutputOptions, PageContext, Pipeline
from manga_translation.artifacts import get_output_path
from manga_translation.steps.page_saving import PageSaving


class PageSavingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.directory = Path(temp_dir.name)
        self.source = self.directory / "page.png"
        Image.new("RGB", (4, 4), "black").save(self.source)

    async def save(self, output_dir: Path) -> PageContext:
        context = PageContext(self.source, {"output_image": Image.new("RGB", (4, 4), "blue")})
        return await Pipeline.from_groups([PageSaving(OutputOptions(output_dir))]).run(context)

    async def test_saves_output_image_into_created_directory(self) -> None:
        output_dir = self.directory / "out" / "nested"
        context = await self.save(output_dir)
        self.assertEqual(get_output_path(context), output_dir / "page.png")
        with Image.open(output_dir / "page.png") as saved:
            self.assertEqual(saved.convert("RGB").getpixel((0, 0)), (0, 0, 255))

    async def test_source_file_is_unchanged(self) -> None:
        before = hashlib.sha256(self.source.read_bytes()).digest()
        await self.save(self.directory / "out")
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).digest(), before)

    async def test_output_path_equal_to_source_is_rejected(self) -> None:
        before = self.source.read_bytes()
        with (
            self.assertLogs("manga_translation.pipeline", "ERROR"),
            self.assertRaisesRegex(ValueError, "would overwrite the source image"),
        ):
            await self.save(self.directory)
        self.assertEqual(self.source.read_bytes(), before)
