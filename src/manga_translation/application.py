"""Assembly of the pipeline and its execution for every input page."""

import asyncio

from .config import AppConfig
from .context import PageContext
from .inputs import collect_image_paths
from .pipeline import Pipeline
from .steps.image_loading import ImageLoading
from .steps.page_saving import PageSaving
from .steps.text_detection import TextDetection
from .steps.text_erasing import TextErasing
from .steps.text_recognition import TextRecognition
from .steps.text_rendering import TextRendering
from .steps.text_translation import TextTranslation


def build_pipeline(config: AppConfig) -> Pipeline:
    """Assemble step groups in processing order; new step groups are added here."""
    return Pipeline.from_groups(
        [
            ImageLoading(config.image),
            TextDetection(config.detection),
            TextRecognition(config.recognition),
            TextTranslation(config.translation),
            TextErasing(config.erasing),
            TextRendering(config.rendering),
            PageSaving(config.output),
        ]
    )


def run(config: AppConfig) -> list[PageContext]:
    """Run the pipeline once per input image and return the final contexts."""
    pipeline = build_pipeline(config)
    return [
        asyncio.run(pipeline.run(PageContext(path)))
        for path in collect_image_paths(config.input_path)
    ]
