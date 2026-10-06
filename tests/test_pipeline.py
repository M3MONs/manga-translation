import asyncio
import logging
import unittest
from pathlib import Path
from unittest.mock import patch

from manga_translation import PageContext, Pipeline
from manga_translation.step import Step, StepGroup


class RecordStep:
    def __init__(self, label: str, events: list[str]) -> None:
        self.label = label
        self.events = events

    async def execute(self, context: PageContext) -> None:
        await asyncio.sleep(0)
        self.events.append(self.label)


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.context = PageContext(Path("page.png"))

    async def test_steps_follow_configured_order(self) -> None:
        events = []
        first = RecordStep("first", events)
        second = RecordStep("second", events)
        await Pipeline([first, second]).run(self.context)
        self.assertEqual(events, ["first", "second"])
        events.clear()
        await Pipeline([second, first]).run(self.context)
        self.assertEqual(events, ["second", "first"])

    async def test_groups_are_flattened_in_order(self) -> None:
        events = []

        class Group(StepGroup):
            def __init__(self, *labels: str) -> None:
                self.labels = labels

            def steps(self) -> list[Step]:
                return [RecordStep(label, events) for label in self.labels]

        await Pipeline.from_groups([Group("a", "b"), Group("c")]).run(self.context)
        self.assertEqual(events, ["a", "b", "c"])

    async def test_pipeline_copies_step_sequence(self) -> None:
        events = []
        steps = [RecordStep("first", events)]
        pipeline = Pipeline(steps)
        steps.clear()
        await pipeline.run(self.context)
        self.assertEqual(events, ["first"])

    async def test_independent_steps_share_context_and_results(self) -> None:
        contexts = []

        class Produce:
            async def execute(self, context: PageContext) -> None:
                contexts.append(context)
                context.artifacts["text"] = "hello"

        class Consume:
            async def execute(self, context: PageContext) -> None:
                contexts.append(context)
                context.artifacts["translation"] = context.artifacts["text"]

        result = await Pipeline([Produce(), Consume()]).run(self.context)
        self.assertIs(result, self.context)
        self.assertTrue(all(context is self.context for context in contexts))
        self.assertEqual(result.artifacts["translation"], "hello")

    async def test_empty_pipeline_returns_original_context(self) -> None:
        with (
            patch("manga_translation.execution_logging.perf_counter", side_effect=[1, 3]),
            self.assertLogs("manga_translation.pipeline", level="INFO") as logs,
        ):
            self.assertIs(await Pipeline([]).run(self.context), self.context)
        self.assertEqual([record.getMessage() for record in logs.records], [
            "Pipeline started page=page.png steps=0",
            "Pipeline completed page=page.png duration=2.000000s",
        ])

    def test_page_artifacts_are_independent(self) -> None:
        other = PageContext(Path("other.png"))
        self.context.artifacts["text"] = "hello"
        self.assertEqual(other.artifacts, {})

    async def test_failure_is_logged_once_and_stops_execution(self) -> None:
        failure = ValueError("failed")
        events = []

        class FailingStep:
            async def execute(self, context: PageContext) -> None:
                context.artifacts["partial"] = True
                raise failure

        with (
            self.assertLogs("manga_translation.pipeline", level="INFO") as logs,
            self.assertRaises(ValueError) as raised,
        ):
            await Pipeline([
                FailingStep(), RecordStep("later", events),
            ]).run(self.context)
        self.assertIs(raised.exception, failure)
        self.assertEqual(events, [])
        self.assertTrue(self.context.artifacts["partial"])
        errors = [record for record in logs.records if record.levelno == logging.ERROR]
        self.assertEqual(len(errors), 1)
        exc_info = errors[0].exc_info
        assert exc_info is not None
        self.assertIs(exc_info[1], failure)
        self.assertIsNotNone(exc_info[2])
        self.assertIn("page=page.png step=1/2 name=", errors[0].getMessage())
        self.assertFalse(any("completed" in record.getMessage() for record in logs.records))

    async def test_logs_describe_step_and_pipeline_duration(self) -> None:
        with (
            patch("manga_translation.execution_logging.perf_counter", side_effect=[1, 2, 4, 6]),
            self.assertLogs("manga_translation.pipeline", level="INFO") as logs,
        ):
            await Pipeline([RecordStep("first", [])]).run(self.context)
        messages = [record.getMessage() for record in logs.records]
        self.assertEqual(len(messages), 4)
        self.assertIn("Pipeline started page=page.png steps=1", messages[0])
        self.assertIn("Step started page=page.png step=1/1 name=RecordStep", messages[1])
        self.assertIn("duration=2.000000s", messages[2])
        self.assertIn("Pipeline completed page=page.png duration=5.000000s", messages[3])

    async def test_cancellation_propagates_and_stops_execution(self) -> None:
        events = []

        class CancelledStep:
            async def execute(self, context: PageContext) -> None:
                raise asyncio.CancelledError

        with (
            self.assertLogs("manga_translation.pipeline", level="INFO") as logs,
            self.assertRaises(asyncio.CancelledError),
        ):
            await Pipeline([CancelledStep(), RecordStep("later", events)]).run(self.context)
        self.assertEqual(events, [])
        self.assertEqual(len(logs.records), 2)
        self.assertTrue(all(record.levelno == logging.INFO for record in logs.records))
        self.assertTrue(all("started" in record.getMessage() for record in logs.records))
