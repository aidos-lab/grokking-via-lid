"""Create and configure a global logger."""

import logging
import pathlib

from grokking.logging.setup_exception_logging import setup_exception_logging


def create_and_configure_global_logger(
    name: str = __name__,
    file: str = __file__,
) -> logging.Logger:
    """Create and configure a global logger."""
    global_logger: logging.Logger = logging.getLogger(
        name=name,
    )
    global_logger.setLevel(
        level=logging.INFO,
    )
    logging_formatter = logging.Formatter(
        fmt="[%(asctime)s][%(levelname)8s][%(name)s] %(message)s (%(filename)s:%(lineno)s)",
    )

    setup_exception_logging(
        logger=global_logger,
    )

    return global_logger
