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

"""Enums used in the configuration classes."""

from enum import IntEnum, StrEnum, auto, unique


class Verbosity(IntEnum):
    """Verbosity level."""

    QUIET = 0
    NORMAL = 1
    VERBOSE = 2
    DEBUG = 3


@unique
class PreferredTorchBackend(StrEnum):
    """The preferred backend for PyTorch."""

    CPU = auto()
    CUDA = auto()
    MPS = auto()
    AUTO = auto()


class NNeighborsMode(StrEnum):
    """The different modes for specifying the number of neighbors in the pointwise estimate computation."""

    ABSOLUTE_SIZE = auto()
    RELATIVE_SIZE = auto()


@unique
class ZeroVectorHandlingMode(StrEnum):
    """The different modes for handling zero vectors."""

    KEEP = auto()
    REMOVE = auto()


@unique
class DeduplicationMode(StrEnum):
    """The different modes for deduplication."""

    IDENTITY = auto()
    ARRAY_DEDUPLICATOR = auto()


@unique
class ArtificialNoiseMode(StrEnum):
    """Different modes for adding artificial noise to the data."""

    DO_NOTHING = auto()
    GAUSSIAN = auto()


# ==============================
# Enums used for local estimates
# ==============================


class EstimatorMethodType(StrEnum):
    """The different types of methods for the estimator."""

    TWONN = auto()
    LPCA = auto()
