# Copyright 2025
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


"""Compute global and local estimates from prepared embeddings."""

import logging
import pprint
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from grokking.analysis.local_estimates_computation.estimator.get_estimator import (
    get_estimator_from_local_estimates_config,
)
from grokking.analysis.local_estimates_computation.get_n_neighbors_from_array_len_and_pointwise_config import (
    get_n_neighbors_from_array_len_and_pointwise_config,
)
from grokking.config_classes.local_estimates.local_estimates_config import LocalEstimatesConfig
from grokking.logging.log_array_info import log_array_info
from grokking.typing.enums import Verbosity

if TYPE_CHECKING:
    from skdim._commonfuncs import GlobalEstimator

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def global_and_pointwise_local_estimates_computation(
    array_for_estimator: np.ndarray,
    local_estimates_config: LocalEstimatesConfig,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> tuple[
    np.ndarray | None,
    np.ndarray,
]:
    """Run the local estimates computation."""
    # # # #
    # Number of neighbors which are used for the computation of the pointwise local estimates
    n_neighbors: int = get_n_neighbors_from_array_len_and_pointwise_config(
        array_len=len(array_for_estimator),
        pointwise_config=local_estimates_config.pointwise,
        verbosity=verbosity,
        logger=logger,
    )

    # # # #
    # Estimator setup
    estimator: GlobalEstimator = get_estimator_from_local_estimates_config(
        local_estimates_config=local_estimates_config,
        verbosity=verbosity,
        logger=logger,
    )

    # # # #
    # Pointwise estimates computation
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg="Pointwise computation.",
        )
        logger.info(
            msg="Calling estimator.fit_pw() ...",
        )

    fitted_pw_estimator = estimator.fit_pw(
        X=array_for_estimator,
        precomputed_knn=None,
        smooth=False,
        n_neighbors=n_neighbors,
        n_jobs=local_estimates_config.pointwise.n_jobs,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg="Calling estimator.fit_pw() DONE",
        )

    pointwise_results_array = list(
        fitted_pw_estimator.dimension_pw_,
    )

    pointwise_results_array_np: np.ndarray = np.array(
        pointwise_results_array,
    )

    if verbosity >= Verbosity.NORMAL:
        log_array_info(
            array_=pointwise_results_array_np,
            array_name="pointwise_results_array_np",
            log_array_size=True,
            log_row_l2_norms=False,  # Note: This is a one-dimensional array, so the l2-norms are not meaningful
            logger=logger,
        )

        # Log the mean and standard deviation of the local estimates
        logger.info(
            msg=f"{pointwise_results_array_np.mean() = }",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"{pointwise_results_array_np.std() = }",  # noqa: G004 - low overhead
        )

    # # # #
    # Global estimate computation
    if local_estimates_config.compute_global_estimates:
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Global computation.",
            )
            logger.info(
                msg="Calling estimator.fit_transform() ...",
            )

        global_dimension = estimator.fit_transform(
            X=array_for_estimator,
        )

        global_estimate_array_np = np.array(
            [global_dimension],
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Calling estimator.fit_transform() DONE",
            )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"global_estimate_array_np:\n{global_estimate_array_np}",  # noqa: G004 - low overhead
            )
    else:
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Skipping global computation.",
            )
        global_estimate_array_np = None

    return (
        global_estimate_array_np,
        pointwise_results_array_np,
    )


def create_additional_pointwise_results_statistics(
    pointwise_results_array_np: np.ndarray,
    truncation_size_range: range = range(
        5_000,
        60_001,
        5_000,
    ),
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> dict:
    """Create additional statistics from the pointwise results array and other computation results."""
    additional_pointwise_results_statistics: dict = {}

    # We collect the statistics of the pointwise results array under a separate key.
    # This allows for a more structured storage of the results and easier extension in the future.
    subkey = "pointwise_results_array_np"
    subdict: dict = make_array_statistics_dict(
        array=pointwise_results_array_np,
        array_name=subkey,
    )

    additional_pointwise_results_statistics[subkey] = subdict

    # Add statistics of truncated pointwise results arrays
    for truncation_size in truncation_size_range:
        subkey: str = f"pointwise_results_array_np_truncated_first_{truncation_size}"
        subdict: dict = make_array_statistics_dict(
            array=pointwise_results_array_np[:truncation_size],
            array_name=subkey,
        )

        additional_pointwise_results_statistics[subkey] = subdict

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"additional_pointwise_results_statistics:\n"  # noqa: G004 - low overhead
            f"{pprint.pformat(object=additional_pointwise_results_statistics)}",
        )

    return additional_pointwise_results_statistics


def make_array_statistics_dict(
    array: np.ndarray,
    array_name: str,
) -> dict:
    """Create a dictionary with statistics about the array."""
    array_statistics_dict: dict = {}

    array_statistics_dict["array_name"] = array_name
    array_statistics_dict["shape"] = array.shape
    array_statistics_dict["np_mean"] = np.mean(
        a=array,
    )
    array_statistics_dict["np_std"] = np.std(
        a=array,
    )

    # Convert into a pandas DataFrame and save the describe() output.
    # Note that numpy and pandas use different versions of the standard deviation,
    # where pandas is the unbiased estimator with N-1 in the denominator,
    # while numpy uses N.
    pd_describe_df: pd.DataFrame = pd.DataFrame(
        data=array,
    ).describe()
    array_statistics_dict["pd_describe"] = pd_describe_df.to_dict()

    return array_statistics_dict
