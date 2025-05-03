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
# Enums used for training
# ==============================


@unique
class LRSchedulerType(StrEnum):
    """Different types of learning rate schedulers."""

    CONSTANT = auto()
    LINEAR = auto()


# ==============================
# Enums used for local estimates
# ==============================


class EstimatorMethodType(StrEnum):
    """The different types of methods for the estimator."""

    TWONN = auto()
    LPCA = auto()
