"""Reading the YAML configuration file and building the application config from it."""

import typing
from pathlib import Path
from typing import Any

import yaml

from .options import (
    AppConfig,
    DetectionOptions,
    ErasingOptions,
    ImageOptions,
    Language,
    OutputOptions,
    RecognitionOptions,
    RenderingOptions,
    TranslationOptions,
)

_TOP_LEVEL_KEYS = {
    "source_language",
    "target_language",
    "image",
    "detection",
    "recognition",
    "translation",
    "erasing",
    "rendering",
    "output",
}
# Settings given once at the top level and passed to every section that needs them.
_SHARED_KEYS = {"source_language", "target_language"}


class ConfigError(ValueError):
    """Raised when configuration data does not describe a valid config."""


def load_config_data(path: Path) -> dict[str, object]:
    """Settings read from a YAML file; an empty file gives no settings."""
    try:
        data = yaml.safe_load(path.read_text("utf-8"))
    except yaml.YAMLError as error:
        raise ConfigError(f"{path}: invalid YAML: {error}") from error
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping of settings")
    return data


def build_config(input_path: Path, data: dict[str, object]) -> AppConfig:
    """Build the config from configuration data; settings left out keep their defaults."""
    unknown = sorted(data.keys() - _TOP_LEVEL_KEYS)
    if unknown:
        raise ConfigError(f"{', '.join(unknown)}: unknown setting")

    source = _language(data, "source_language")
    target = _language(data, "target_language")

    return AppConfig(
        input_path,
        image=ImageOptions(**_section(data, "image", ImageOptions)),
        detection=DetectionOptions(**_section(data, "detection", DetectionOptions)),
        recognition=RecognitionOptions(**source, **_section(data, "recognition", RecognitionOptions)),
        translation=TranslationOptions(**source, **target, **_section(data, "translation", TranslationOptions)),
        erasing=ErasingOptions(**_section(data, "erasing", ErasingOptions)),
        rendering=RenderingOptions(**target, **_section(data, "rendering", RenderingOptions)),
        output=OutputOptions(**_section(data, "output", OutputOptions)),
    )


def _language(data: dict[str, object], key: str) -> dict[str, Any]:
    """The language setting as keyword arguments; empty when it is not given."""
    if key not in data:
        return {}
    return {key: _convert(key, Language, data[key])}


def _section(data: dict[str, object], name: str, options: type) -> dict[str, Any]:
    """Settings of one section, converted to the field types of its option class."""
    values = data.get(name) or {}
    if not isinstance(values, dict):
        raise ConfigError(f"{name}: expected a mapping of settings")

    field_types = typing.get_type_hints(options)
    for key in values:
        if key in _SHARED_KEYS:
            raise ConfigError(f"{name}.{key}: set it at the top level")
        if key not in field_types:
            raise ConfigError(f"{name}.{key}: unknown setting")

    return {key: _convert(f"{name}.{key}", field_types[key], value) for key, value in values.items()}


def _convert(name: str, field_type: type, value: object) -> object:
    """Value converted to field_type; bools are not accepted as numbers."""
    if field_type is Path and isinstance(value, str):
        return Path(value)
    if field_type is Language and isinstance(value, str) and value in Language:
        return Language(value)
    if field_type is int and isinstance(value, int) and not isinstance(value, bool):
        return value
    if field_type is float and isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)

    expected = "one of " + ", ".join(Language) if field_type is Language else field_type.__name__
    raise ConfigError(f"{name}: expected {expected}, got {value!r}")
