"""Sequential orchestration of asynchronous steps."""

from collections.abc import Sequence

from ..logs.execution import LoggedStep, log_pipeline
from .context import PageContext
from .step import Step, StepGroup


class Pipeline:
    def __init__(self, steps: Sequence[Step]) -> None:
        configured_steps = tuple(steps)
        self._steps = tuple(
            LoggedStep(step, position, len(configured_steps))
            for position, step in enumerate(configured_steps, start=1)
        )

    @classmethod
    def from_groups(cls, groups: Sequence[StepGroup]) -> "Pipeline":
        """Flatten step groups, in order, into one pipeline."""
        return cls([step for group in groups for step in group.steps()])

    @log_pipeline
    async def run(self, context: PageContext) -> PageContext:
        for step in self._steps:
            await step.execute(context)
        return context
