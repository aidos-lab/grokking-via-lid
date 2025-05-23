# Copyright 2024-2025
# [ANONYMIZED_INSTITUTION],
# [ANONYMIZED_FACULTY],
# [ANONYMIZED_DEPARTMENT]
#
# Authors:
# AUTHOR_1 (2025) (author1@example.com)
# AUTHOR_2 (author2@example.com)
#
# Code generation tools and workflows:
# First versions of this code were potentially generated
# with the help of AI writing assistants including
# GitHub Copilot, ChatGPT, Microsoft Copilot, Google Gemini.
# Afterwards, the generated segments were manually reviewed and edited.
#


"""Configurations for adding artificial noise into the local estimates computation."""

from pydantic import BaseModel, Field

from grokking.config_classes.constants import ITEM_SEP, KV_SEP, NAME_PREFIXES
from grokking.typing.enums import ArtificialNoiseMode


class LocalEstimatesNoiseConfig(BaseModel):
    """Configurations for adding artificial noise into the local estimates computation."""

    artificial_noise_mode: ArtificialNoiseMode = Field(
        default=ArtificialNoiseMode.DO_NOTHING,
        title="Artificial noise mode.",
        description="Which kind of noise will be added before the local estimates computation.",
    )

    distortion_parameter: float = Field(
        default=0.0,
        title="Distortion parameter.",
        description="For example, this will be the standard deviation of the Gaussian noise.",
    )

    seed: int = Field(
        default=0,
        title="Noise seed.",
        description="The random seed for the noise generation.",
    )

    @property
    def config_description(
        self,
    ) -> str:
        """Get the description of the config."""
        match self.artificial_noise_mode:
            case ArtificialNoiseMode.DO_NOTHING:
                description = (
                    f"{NAME_PREFIXES['local_estimates_noise_artificial_noise_mode']}"
                    + KV_SEP
                    + f"{str(object=self.artificial_noise_mode)}"
                )
            case ArtificialNoiseMode.GAUSSIAN:
                description = (
                    f"{NAME_PREFIXES['local_estimates_noise_artificial_noise_mode']}"
                    + KV_SEP
                    + f"{str(object=self.artificial_noise_mode)}"
                    + ITEM_SEP
                    + f"{NAME_PREFIXES['local_estimates_noise_distortion_parameter']}"
                    + KV_SEP
                    + f"{str(object=self.distortion_parameter)}"
                    + ITEM_SEP
                    + f"{NAME_PREFIXES['local_estimates_noise_seed']}"
                    + KV_SEP
                    + f"{str(object=self.seed)}"
                )
            case _:
                msg: str = f"Unknown {self.artificial_noise_mode = }"
                raise ValueError(
                    msg,
                )

        return description
