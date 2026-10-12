"""Data shared by the steps processing one page."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PageContext:
    source_path: Path
    artifacts: dict[str, object] = field(default_factory=dict)
