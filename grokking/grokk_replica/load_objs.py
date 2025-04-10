"""Loading objects from config files."""

import logging

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
    name,
):
    def add_f(f):
        registry[name] = f
        return f

    return add_f


def load_item(
    config,
    *args,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
):
    config = config.copy()
    name = config.pop("name")
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


@register("mod_sum_dataset")
def load_mod_sum_dataset(
    config,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
):
    return ModSumDataset(
        p=config["p"],
        frac_train=config["frac_train"],
    )


@register("mod_subtract_dataset")
def load_mod_subtract_dataset(
    config,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
):
    return ModSubtractDataset(
        p=config["p"],
        frac_train=config["frac_train"],
    )


@register("mod_division_dataset")
def load_mod_division_dataset(
    config,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
):
    return ModDivisonDataset(
        p=config["p"],
        frac_train=config["frac_train"],
    )


@register("permutation_group_dataset")
def load_mod_permutation_dataset(
    config,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
):
    return PermutationGroup(
        k=config["k"],
        frac_train=config["frac_train"],
    )


@register("grokk_model")
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
                convert_path(config["checkpoint_path"]),  # type: ignore - problem with typing inference of the key
                map_location="cpu",
            ),
            strict=config["strict_load"],
        )
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Checkpoint loaded.",
            )

    return model
