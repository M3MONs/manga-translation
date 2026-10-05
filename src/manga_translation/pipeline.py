"""Sequential orchestration of asynchronous steps."""

from collections.abc import Sequence

from .context import PageContext
from .execution_logging import LoggedStep, log_pipeline
from .step import Step


class Pipeline:
    def __init__(self, steps: Sequence[Step]) -> None:
        configured_steps = tuple(steps)
        self._steps = tuple(
            LoggedStep(step, position, len(configured_steps))
            for position, step in enumerate(configured_steps, start=1)
        )

    @log_pipeline
    async def run(self, context: PageContext) -> PageContext:
        for step in self._steps:
            await step.execute(context)
        return context
