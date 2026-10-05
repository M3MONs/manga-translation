"""Execution lifecycle logging for pipeline steps and runs."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Coroutine
from functools import wraps
from time import perf_counter
from typing import TYPE_CHECKING, Any

from .context import PageContext
from .step import Step

if TYPE_CHECKING:
    from .pipeline import Pipeline

logger = logging.getLogger("manga_translation.pipeline")


class LoggedStep:
    def __init__(self, step: Step, position: int, total: int) -> None:
        self._step = step
        self._position = position
        self._total = total

    async def execute(self, context: PageContext) -> None:
        name = type(self._step).__qualname__
        started = perf_counter()
        details = (context.source_path, self._position, self._total, name)
        logger.info(
            "Step started page=%s step=%d/%d name=%s", *details,
        )
        try:
            await self._step.execute(context)
        except Exception:
            logger.exception(
                "Step failed page=%s step=%d/%d name=%s duration=%.6fs",
                *details, perf_counter() - started,
            )
            raise
        logger.info(
            "Step completed page=%s step=%d/%d name=%s duration=%.6fs",
            *details, perf_counter() - started,
        )


def log_pipeline(
    run: Callable[[Pipeline, PageContext], Awaitable[PageContext]],
) -> Callable[[Pipeline, PageContext], Coroutine[Any, Any, PageContext]]:
    @wraps(run)
    async def logged_run(self: Pipeline, context: PageContext) -> PageContext:
        started = perf_counter()
        logger.info(
            "Pipeline started page=%s steps=%d", context.source_path, len(self._steps)
        )
        result = await run(self, context)
        logger.info(
            "Pipeline completed page=%s duration=%.6fs",
            context.source_path, perf_counter() - started,
        )
        return result

    return logged_run
