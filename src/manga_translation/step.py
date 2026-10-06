"""Contracts implemented by independent pipeline steps and their groups."""

from abc import ABC, abstractmethod
from typing import Protocol

from .context import PageContext


class Step(Protocol):
    async def execute(self, context: PageContext) -> None:
        """Read required artifacts and add results to the page context."""
        ...


class StepGroup(ABC):
    """Ordered steps configured from one options object."""

    @abstractmethod
    def steps(self) -> list[Step]:
        """Return the group's steps in processing order."""
