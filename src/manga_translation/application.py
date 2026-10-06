"""Assembly of the pipeline and its execution for every input page."""

import asyncio

from .config import AppConfig
from .context import PageContext
from .inputs import collect_image_paths
from .pipeline import Pipeline
from .steps.image_loading import ImageLoading


def build_pipeline(config: AppConfig) -> Pipeline:
    """Assemble step groups in processing order; new step groups are added here."""
    return Pipeline.from_groups([ImageLoading(config.image)])


def run(config: AppConfig) -> list[PageContext]:
    """Run the pipeline once per input image and return the final contexts."""
    pipeline = build_pipeline(config)
    return [
        asyncio.run(pipeline.run(PageContext(path)))
        for path in collect_image_paths(config.input_path)
    ]
