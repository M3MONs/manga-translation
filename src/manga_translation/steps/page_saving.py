"""Steps that write the translated page to a new file, leaving the source untouched."""

from ..artifacts import get_output_image
from ..config import OutputOptions
from ..context import PageContext
from ..step import Step, StepGroup


class SavePage:
    """Requires output_image and source_path; produces output_path."""

    def __init__(self, options: OutputOptions) -> None:
        self.options = options

    async def execute(self, context: PageContext) -> None:
        path = self.options.output_dir / context.source_path.name
        if path.resolve() == context.source_path.resolve():
            raise ValueError(f"Output path {path} would overwrite the source image")
        path.parent.mkdir(parents=True, exist_ok=True)
        get_output_image(context).save(path)
        context.artifacts["output_path"] = path


class PageSaving(StepGroup):
    """Save the page into the output directory under its source file name."""

    def __init__(self, options: OutputOptions) -> None:
        self.options = options

    def steps(self) -> list[Step]:
        return [SavePage(self.options)]
