import logging
import os
import pathlib
from itertools import product

import torch
from tqdm.auto import tqdm

import wandb
from grokking.analysis.local_estimates_computation.global_and_pointwise_local_estimates_computation import (
    create_additional_pointwise_results_statistics,
    global_and_pointwise_local_estimates_computation,
)
from grokking.config_classes.local_estimates.filtering_config import LocalEstimatesFilteringConfig
from grokking.config_classes.local_estimates.local_estimates_config import LocalEstimatesConfig
from grokking.config_classes.local_estimates.plot_config import LocalEstimatesPlotConfig, PlotSavingConfig
from grokking.config_classes.local_estimates.pointwise_config import LocalEstimatesPointwiseConfig
from grokking.grokk_replica.grokk_model import GrokkModel
from grokking.scripts.collect_hidden_states import collect_hidden_states, preprocess_hidden_states
from grokking.scripts.dataset_for_topological_analysis import (
    DatasetForTopologicalAnalysis,
)
from grokking.scripts.generate_tsne_visualizations import generate_tsne_visualizations
from grokking.scripts.input_and_hidden_states_array import InputAndHiddenStatesArray
from grokking.typing.enums import DeduplicationMode, NNeighborsMode, TokenRestrictionMode, Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)
default_device: torch.device = torch.device(
    device="cpu",
)


def do_topological_analysis_step(
    datasets_for_topological_analysis_list: list[DatasetForTopologicalAnalysis],
    model: GrokkModel,
    output_dir: os.PathLike,
    topological_analysis_cfg: dict,
    step: int,
    topological_analysis_create_projection_plot_every: int,
    *,
    use_wandb: bool = False,
    device: torch.device = default_device,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"{step + 1 = }: Running topological analysis ...",  # noqa: G004 - low overhead
        )

    if verbosity >= Verbosity.DEBUG:
        logger.debug(
            msg="Setting model to eval mode.",
        )
    model.eval()

    with torch.no_grad():
        for dataset_for_topological_analysis in datasets_for_topological_analysis_list:
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg=f"{step + 1 = }: Running topological analysis for "  # noqa: G004 - low overhead
                    f"{dataset_for_topological_analysis.split = } ...",
                )

            token_restriction_mode_choices: list[TokenRestrictionMode] = topological_analysis_cfg[
                "token_restriction_mode_choices"
            ]
            make_copy_of_dataloader: bool = topological_analysis_cfg["make_copy_of_dataloader"]

            for token_restriction_mode in token_restriction_mode_choices:
                logger.info(
                    msg=f"{step + 1 = }: Running analysis with {token_restriction_mode = } ...",  # noqa: G004 - low overhead
                )

                if verbosity >= Verbosity.NORMAL:
                    logger.info(
                        msg=f"{step + 1 = }: Collecting hidden states ...",  # noqa: G004 - low overhead
                    )

                input_and_hidden_states_array: InputAndHiddenStatesArray = collect_hidden_states(
                    model=model,
                    topological_analysis_cfg=topological_analysis_cfg,
                    dataset_for_topological_analysis=dataset_for_topological_analysis,
                    token_restriction_mode=token_restriction_mode,
                    make_copy_of_dataloader=make_copy_of_dataloader,
                    device=device,
                    verbosity=verbosity,
                    logger=logger,
                )

                if verbosity >= Verbosity.NORMAL:
                    logger.info(
                        msg=f"{step + 1 = }: Collecting hidden states DONE",  # noqa: G004 - low overhead
                    )

                # # # #
                # Preprocess the hidden states
                topo_sampling_seed: int = topological_analysis_cfg["sampling_seed"]
                number_of_samples_choices: list[int] = topological_analysis_cfg["number_of_samples_choices"]
                absolute_n_neighbors_choices: list[int] = topological_analysis_cfg["absolute_n_neighbors_choices"]

                # Note: This list can be extended with more parameters
                local_estimates_parameters_combinations = product(
                    number_of_samples_choices,
                    absolute_n_neighbors_choices,
                )

                for (
                    number_of_samples,
                    absolute_n_neighbors,
                ) in tqdm(
                    local_estimates_parameters_combinations,
                    desc=f"{step + 1 = }: Iterating over parameters for local estimates.",
                ):
                    if verbosity >= Verbosity.NORMAL:
                        logger.info(
                            msg=f"{step + 1 = }: Preprocessing hidden states ...",  # noqa: G004 - low overhead
                        )

                    local_estimates_config = LocalEstimatesConfig(
                        filtering=LocalEstimatesFilteringConfig(
                            num_samples=number_of_samples,
                            deduplication_mode=DeduplicationMode.ARRAY_DEDUPLICATOR,
                        ),
                        pointwise=LocalEstimatesPointwiseConfig(
                            n_neighbors_mode=NNeighborsMode.ABSOLUTE_SIZE,
                            absolute_n_neighbors=absolute_n_neighbors,
                        ),
                    )

                    input_and_hidden_states_array_preprocessed: InputAndHiddenStatesArray = preprocess_hidden_states(
                        input_and_hidden_states_array=input_and_hidden_states_array,
                        topo_sampling_seed=topo_sampling_seed,
                        local_estimates_config=local_estimates_config,
                        verbosity=verbosity,
                        logger=logger,
                    )

                    # Example:
                    # (This examples discusses the only operand case)
                    #
                    # For p=96, there are 96 * 96 = 9216 possible pairs of elements.
                    # Since we have an autoregressive model, the state of the first operand is independent of the context,
                    # thus we have 96 different hidden states for the first operand.
                    # For the second operand, we have 96 * 96 = 9216 different hidden states.
                    #
                    # Thus for a train portion of 0.4, we have
                    # 96 * 96 * 0.4 + 96 = 9216 * 0.4 + 96 = 3686.4 + 96 = 3782.4
                    # different train hidden states.
                    # > self.hidden_states.shape=(3782, 128)
                    #
                    # For validation, we have
                    # 96 * 96 * 0.6 + 96 = 9216 * 0.6 + 96 = 5529.6 + 96 = 5625.6
                    # > self.hidden_states.shape=(5626, 128)
                    #
                    # Note that the number of hidden states is rounded up or down to the nearest integer.

                    if verbosity >= Verbosity.NORMAL:
                        logger.info(
                            msg=f"{step + 1 = }: Preprocessing hidden states DONE",  # noqa: G004 - low overhead
                        )

                    # # # #
                    # Analyse the extracted hidden states

                    (
                        global_estimate_array_np,
                        pointwise_results_array_np,
                    ) = global_and_pointwise_local_estimates_computation(
                        array_for_estimator=input_and_hidden_states_array_preprocessed.hidden_states,
                        local_estimates_config=local_estimates_config,
                        verbosity=verbosity,
                        logger=logger,
                    )

                    # Note: Currently, these statistics are not used or logged
                    additional_pointwise_results_statistics: dict = create_additional_pointwise_results_statistics(
                        pointwise_results_array_np=pointwise_results_array_np,
                        truncation_size_range=range(500, 5_000, 500),
                        verbosity=Verbosity.QUIET,  # We do not want to clutter the logfile with this information
                        logger=logger,
                    )

                    topological_estimates_dict: dict = {
                        dataset_for_topological_analysis.split: {
                            token_restriction_mode: {
                                f"{local_estimates_config.config_description}": {
                                    f"{local_estimates_config.pointwise.config_description}": {
                                        "mean": pointwise_results_array_np.mean(),
                                        "std": pointwise_results_array_np.std(),
                                    },
                                    # The global estimate does not depend on the pointwise config.
                                    # If available, the global estimate array contains only a single element.
                                    "global": global_estimate_array_np[0]
                                    if global_estimate_array_np is not None
                                    else None,
                                },
                            },
                        },
                        "step": (step + 1),
                    }

                    if use_wandb:
                        wandb.log(
                            data=topological_estimates_dict,
                        )

                        if verbosity >= Verbosity.NORMAL:
                            logger.info(
                                msg="Logged local estimates results to wandb.",
                            )

                    # # # #
                    # Note:
                    # - If you want to save the local estimates results to a file, this would be the place to do it.

                    # # # #
                    # Optional plotting
                    if (
                        topological_analysis_create_projection_plot_every > 0
                        and (step + 1) % topological_analysis_create_projection_plot_every == 0
                    ):
                        if verbosity >= Verbosity.NORMAL:
                            logger.info(
                                msg="Creating projection plot ...",
                            )

                        local_estimates_plot_config = LocalEstimatesPlotConfig(
                            pca_n_components=None,  # Skip the PCA step
                            saving=PlotSavingConfig(
                                save_html=False,  # Since the .html is quite large, we skip saving it for now
                                save_pdf=True,
                                save_csv=True,
                            ),
                        )

                        saved_plots_local_estimates_root_dir = pathlib.Path(
                            output_dir,
                            "plots",
                            "local_estimates_projection",
                            f"{token_restriction_mode=}",
                            local_estimates_config.config_description,
                            local_estimates_config.pointwise.config_description,
                            f"{step+1=}",
                            f"{dataset_for_topological_analysis.split=}",
                        )

                        generate_tsne_visualizations(
                            input_and_hidden_states_array=input_and_hidden_states_array_preprocessed,
                            pointwise_results_array_np=pointwise_results_array_np,
                            local_estimates_plot_config=local_estimates_plot_config,
                            saved_plots_root_dir=saved_plots_local_estimates_root_dir,
                            verbosity=verbosity,
                            logger=logger,
                        )

                        if verbosity >= Verbosity.NORMAL:
                            logger.info(
                                msg="Creating projection plot DONE",
                            )

                logger.info(
                    msg=f"{step + 1 = }: Running analysis with {token_restriction_mode = } ...",  # noqa: G004 - low overhead
                )

            logger.info(
                msg=f"{step + 1 = }: Running topological analysis for {dataset_for_topological_analysis.split = } DONE",  # noqa: G004 - low overhead
            )

    if verbosity >= Verbosity.DEBUG:
        logger.debug(
            msg="Setting model to train mode.",
        )
    model.train()

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"{step + 1 = }: Running topological analysis DONE",  # noqa: G004 - low overhead
        )
