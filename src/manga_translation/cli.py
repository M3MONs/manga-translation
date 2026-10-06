"""Command-line entry point translating a page or a directory of pages."""

from pathlib import Path
from typing import Annotated

import typer

from .application import run
from .config import AppConfig, ImageOptions
from .inputs import InputError
from .logging_config import configure_logging

app = typer.Typer()


@app.command()
def translate(
    input_path: Annotated[Path, typer.Argument(exists=True, help="Image file or directory of images.")],
    max_size: Annotated[int, typer.Option(min=0, help="Longest image side in pixels; 0 disables resizing.")] = 4096,
) -> None:
    configure_logging()
    try:
        contexts = run(AppConfig(input_path, ImageOptions(max_size)))
    except InputError as error:
        raise typer.BadParameter(str(error)) from error
    for context in contexts:
        typer.echo(f"{context.source_path}: done")
