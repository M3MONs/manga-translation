"""Contract implemented by independent pipeline steps."""

from typing import Protocol

from .context import PageContext


class Step(Protocol):
    async def execute(self, context: PageContext) -> None:
        """Read required artifacts and add results to the page context."""
        ...
