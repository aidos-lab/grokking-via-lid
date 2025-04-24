# Copyright 2025
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

"""Sanitize and truncate directory name."""

import logging

from grokking.config_classes.constants import FILE_NAME_TRUNCATION_LENGTH

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def truncate_length_of_desc(
    desc: str,
    truncation_length: int = FILE_NAME_TRUNCATION_LENGTH,
    logger: logging.Logger = default_logger,
) -> str:
    """Truncate the length of the description string if it is too long."""
    if len(desc) > truncation_length:
        logger.warning(
            msg=f"Too long:\n{desc = }\n{len(desc) = }\nWill be truncated to {truncation_length = } characters.",  # noqa: G004 - low overhead
        )
        desc = desc[:truncation_length]

    return desc


def sanitize_dirname(
    dir_name: str,
) -> str:
    """Sanitizes a directory name by replacing all slashes with underscores.

    Args:
    ----
        dir_name: The directory name to sanitize.

    Returns:
    -------
        The sanitized directory name.

    """
    result = dir_name.replace(
        "/",
        "_",
    ).replace(
        "\\",
        "_",
    )

    result: str = truncate_length_of_desc(
        desc=result,
    )

    if len(result) == 0:
        result = "no_overrides"

    return result
