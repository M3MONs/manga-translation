"""Command-line entry point translating a page or a directory of pages."""

from pathlib import Path
from typing import Annotated

import typer

from .application import run
from .config import (
    AppConfig,
    ImageOptions,
    Language,
    OutputOptions,
    RenderingOptions,
    TranslationOptions,
)
from .inputs import InputError
from .logging_config import configure_logging

app = typer.Typer()


@app.command()
def translate(
    input_path: Annotated[Path, typer.Argument(exists=True, help="Image file or directory of images.")],
    max_size: Annotated[int, typer.Option(min=0, help="Longest image side in pixels; 0 disables resizing.")] = 4096,
    language: Annotated[Language, typer.Option(help="Target language of the translation.")] = Language.EN,
    output: Annotated[Path, typer.Option(help="Directory for the translated pages.")] = Path("output"),
) -> None:
    configure_logging()
    config = AppConfig(
        input_path,
        ImageOptions(max_size),
        translation=TranslationOptions(language),
        rendering=RenderingOptions(language),
        output=OutputOptions(output),
    )
    try:
        contexts = run(config)
    except InputError as error:
        raise typer.BadParameter(str(error)) from error
    for context in contexts:
        typer.echo(f"{context.source_path}: done")
