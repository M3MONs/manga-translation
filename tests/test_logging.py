import logging
import subprocess
import sys
import unittest
from io import StringIO
from unittest.mock import patch

from manga_translation.logging_config import configure_logging


class LoggingTests(unittest.TestCase):
    def setUp(self) -> None:
        root = logging.getLogger()
        self.handlers = root.handlers[:]
        self.level = root.level

    def tearDown(self) -> None:
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)
            handler.close()
        root.handlers[:] = self.handlers
        root.setLevel(self.level)

    def test_reconfiguration_outputs_each_event_once_on_stderr(self) -> None:
        stream = StringIO()
        with patch("sys.stderr", stream):
            configure_logging()
            configure_logging()
            logging.getLogger("manga_translation.pipeline").info("unique event")
        lines = stream.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertIn("INFO manga_translation.pipeline unique event", lines[0])

    def test_configurable_level_and_existing_logger_remain_enabled(self) -> None:
        existing = logging.getLogger("existing.application")
        stream = StringIO()
        with patch("sys.stderr", stream):
            configure_logging("DEBUG")
            existing.debug("debug event")
        self.assertFalse(existing.disabled)
        self.assertIn("DEBUG existing.application debug event", stream.getvalue())

    def test_default_level_filters_debug(self) -> None:
        stream = StringIO()
        with patch("sys.stderr", stream):
            configure_logging()
            logging.getLogger("manga_translation.pipeline").debug("hidden")
        self.assertEqual(stream.getvalue(), "")

    def test_import_does_not_configure_logging(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c", (
                "import logging; "
                "root = logging.getLogger(); "
                "handlers = root.handlers[:]; level = root.level; "
                "import manga_translation; "
                "import manga_translation.logging_config; "
                "assert root.handlers == handlers; assert root.level == level"
            )],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
