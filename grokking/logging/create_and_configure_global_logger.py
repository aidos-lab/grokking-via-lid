# Copyright 2025
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Matthias Ruppik (2025) (mail@ruppik.net)
#
# Code generation tools and workflows:
# First versions of this code were potentially generated
# with the help of AI writing assistants including
# GitHub Copilot, ChatGPT, Microsoft Copilot, Google Gemini.
# Afterwards, the generated segments were manually reviewed and edited.
#


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
