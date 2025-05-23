# Copyright 2025
# [ANONYMIZED_INSTITUTION],
# [ANONYMIZED_FACULTY],
# [ANONYMIZED_DEPARTMENT]
#
# Authors:
# Charlie Snell (2022)
# AUTHOR_1 (2025) (author1@example.com)
#
# Code generation tools and workflows:
# First versions of this code were potentially generated
# with the help of AI writing assistants including
# GitHub Copilot, ChatGPT, Microsoft Copilot, Google Gemini.
# Afterwards, the generated segments were manually reviewed and edited.
#


"""Loading objects from config files."""

import logging
from collections.abc import Callable

from grokking.grokk_replica.datasets import (
    ModDivisonDataset,
    ModMultiplyDataset,
    ModSubtractDataset,
    ModSumDataset,
    PermutationGroup,
)
from grokking.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)

registry: dict = {}


def register(
    name: str,
) -> Callable:
    def add_f(f):
        registry[name] = f
        return f

    return add_f


def load_item(
    config: dict,
    *args,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
):
    config_copy: dict = config.copy()
    name = config_copy.pop(
        "name",
    )
    if name not in registry:
        msg: str = f"{name=} is not in the registry. Available names are: {registry.keys()}"
        raise NotImplementedError(
            msg,
        )

    # The name will determine the class to be loaded
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Loading {name = }:\n{config_copy = }",  # noqa: G004 - low overhead
        )
    return registry[name](
        config_copy,
        *args,
        verbosity=verbosity,
        logger=logger,
    )


@register(name="mod_sum_dataset")
def load_mod_sum_dataset(
    config: dict,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> ModSumDataset:
    return ModSumDataset(
        p=config["p"],
        frac_train=config["frac_train"],
        dataset_seed=config["dataset_seed"],
    )


@register(name="mod_subtract_dataset")
def load_mod_subtract_dataset(
    config: dict,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> ModSubtractDataset:
    return ModSubtractDataset(
        p=config["p"],
        frac_train=config["frac_train"],
        dataset_seed=config["dataset_seed"],
    )


@register(name="mod_multiply_dataset")
def load_mod_multiply_dataset(
    config: dict,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> ModMultiplyDataset:
    return ModMultiplyDataset(
        p=config["p"],
        frac_train=config["frac_train"],
        dataset_seed=config["dataset_seed"],
    )


@register(name="mod_division_dataset")
def load_mod_division_dataset(
    config: dict,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> ModDivisonDataset:
    return ModDivisonDataset(
        p=config["p"],
        frac_train=config["frac_train"],
        dataset_seed=config["dataset_seed"],
    )


@register(name="permutation_group_dataset")
def load_mod_permutation_dataset(
    config: dict,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> PermutationGroup:
    return PermutationGroup(
        k=config["k"],
        frac_train=config["frac_train"],
        dataset_seed=config["dataset_seed"],
    )
