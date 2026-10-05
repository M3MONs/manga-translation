"""Run a pipeline example without reading an image or contacting services."""

import asyncio
from pathlib import Path

from manga_translation import PageContext, Pipeline
from manga_translation.logging_config import configure_logging


class IdentifyPage:
    """Requires source_path; produces the page_name artifact."""

    async def execute(self, context: PageContext) -> None:
        context.artifacts["page_name"] = context.source_path.stem


class DescribePage:
    """Requires page_name; produces the description artifact."""

    async def execute(self, context: PageContext) -> None:
        context.artifacts["description"] = f"Example page: {context.artifacts['page_name']}"


def main() -> None:
    configure_logging()
    pipeline = Pipeline([IdentifyPage(), DescribePage()])
    context = asyncio.run(pipeline.run(PageContext(Path("example-page.png"))))
    print(context.artifacts["description"])


if __name__ == "__main__":
    main()
