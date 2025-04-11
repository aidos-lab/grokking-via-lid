# Copyright 2025
#
# Authors:
# Charlie Snell (2022)
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

"""Loading objects from config files."""

import logging
from collections.abc import Callable

import torch

from grokking.grokk_replica.datasets import (
    ModDivisonDataset,
    ModSubtractDataset,
    ModSumDataset,
    PermutationGroup,
)
from grokking.grokk_replica.grokk_model import GrokkModel
from grokking.grokk_replica.utils import convert_path
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
    config = config.copy()
    name = config.pop(
        "name",
    )
    if name not in registry:
        raise NotImplementedError
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Loading {name = }:\n{config = }",  # noqa: G004 - low overhead
        )
    return registry[name](
        config,
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


@register(name="grokk_model")
def load_grokk_model(
    config,
    vocab_size,
    out_size,
    device,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> GrokkModel:
    model: GrokkModel = GrokkModel(
        transformer_config=config["transformer_config"],
        vocab_size=vocab_size,
        output_size=out_size,
        device=device,
    ).to(
        device=device,
    )
    if config["checkpoint_path"] is not None:
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"loading grokk_model state dict from: {convert_path(config['checkpoint_path'])}",  # noqa: G004 - low overhead
            )
        model.load_state_dict(
            state_dict=torch.load(
                f=convert_path(config["checkpoint_path"]),  # type: ignore - problem with typing inference of the key
                map_location="cpu",
            ),
            strict=config["strict_load"],
        )
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Checkpoint loaded.",
            )

    return model
