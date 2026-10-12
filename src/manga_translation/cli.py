"""Command-line entry point translating a page or a directory of pages."""

from dataclasses import replace
from pathlib import Path
from typing import Annotated

import typer

from .application import run
from .config import AppConfig, Language
from .config_file import build_config, load_config_data
from .inputs import InputError
from .logging_config import configure_logging

DEFAULT_CONFIG_PATH = Path("config.yaml")

app = typer.Typer()


@app.command()
def translate(
    input_path: Annotated[Path, typer.Argument(exists=True, help="Image file or directory of images.")],
    config: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, help="YAML settings file; defaults to ./config.yaml if it exists."),
    ] = None,
    max_size: Annotated[int | None, typer.Option(min=0, help="Longest image side in pixels; 0 disables resizing.")] = None,
    language: Annotated[Language | None, typer.Option(help="Target language of the translation.")] = None,
    output: Annotated[Path | None, typer.Option(help="Directory for the translated pages.")] = None,
) -> None:
    configure_logging()
    app_config = _load_config(input_path, config)
    app_config = _apply_options(app_config, max_size, language, output)
    try:
        contexts = run(app_config)
    except InputError as error:
        raise typer.BadParameter(str(error)) from error
    for context in contexts:
        typer.echo(f"{context.source_path}: done")


def _load_config(input_path: Path, config_path: Path | None) -> AppConfig:
    """Config built from the settings file, or from the defaults when there is none."""
    if config_path is None and DEFAULT_CONFIG_PATH.is_file():
        config_path = DEFAULT_CONFIG_PATH
    try:
        data = load_config_data(config_path) if config_path else {}
        return build_config(input_path, data)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--config") from error


def _apply_options(config: AppConfig, max_size: int | None, language: Language | None, output: Path | None) -> AppConfig:
    """Config with the command-line options that were given replacing the file settings."""
    if max_size is not None:
        config = replace(config, image=replace(config.image, max_size=max_size))
    if language is not None:
        config = replace(
            config,
            translation=replace(config.translation, target_language=language),
            rendering=replace(config.rendering, target_language=language),
        )
    if output is not None:
        config = replace(config, output=replace(config.output, output_dir=output))
    return config
