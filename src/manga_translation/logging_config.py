"""Application logging setup; importing the package has no logging side effects."""

from logging.config import dictConfig


def configure_logging(level: str = "INFO") -> None:
    """Configure text logs on stderr without duplicating root handlers."""
    dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "text": {
                "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
                "formatter": "text",
            },
        },
        "root": {"level": level, "handlers": ["console"]},
    })
