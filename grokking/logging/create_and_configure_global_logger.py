# Copyright 2025
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (mail@ruppik.net)
#
# Code generation tools and workflows:
# First versions of this code were potentially generated
# with the help of AI writing assistants including
# GitHub Copilot, ChatGPT, Microsoft Copilot, Google Gemini.
# Afterwards, the generated segments were manually reviewed and edited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
