"""Logging setup with rich terminal output"""

import logging
from rich.logging import RichHandler


def get_logger(name: str) -> logging.Logger:
    """Configure and return a logger with rich console output"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    return logging.getLogger(name)
