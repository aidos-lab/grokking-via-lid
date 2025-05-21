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


import logging
import pprint

import skdim

from grokking.config_classes.local_estimates.local_estimates_config import LocalEstimatesConfig
from grokking.typing.enums import EstimatorMethodType, Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def get_estimator_from_local_estimates_config(
    local_estimates_config: LocalEstimatesConfig,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> skdim._commonfuncs.GlobalEstimator:
    """Get the estimator from the local estimates configuration."""
    match local_estimates_config.estimator.method_type:
        case EstimatorMethodType.TWONN:
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg="Using the TwoNN estimator.",
                )
            estimator = skdim.id.TwoNN(
                discard_fraction=local_estimates_config.estimator.twonn_discard_fraction,
            )
        case EstimatorMethodType.LPCA:
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg="Using lPCA estimator.",
                )
            estimator = skdim.id.lPCA(
                ver=local_estimates_config.estimator.lpca_ver,
                alphaRatio=local_estimates_config.estimator.lpca_alphaRatio,
                alphaFO=local_estimates_config.estimator.lpca_alphaFO,
                alphaFan=local_estimates_config.estimator.lpca_alphaFan,
                betaFan=local_estimates_config.estimator.lpca_betaFan,
                PFan=local_estimates_config.estimator.lpca_PFan,
            )
        # Note: You can add additional estimators here.
        case _:
            msg: str = f"Unsupported estimator method type: {local_estimates_config.estimator.method_type =}"
            logger.error(
                msg=msg,
            )
            raise ValueError(
                msg,
            )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"estimator:\n{estimator}",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"estimator.__dict__:\n{pprint.pformat(object=estimator.__dict__)}",  # noqa: G004 - low overhead
        )

    return estimator
