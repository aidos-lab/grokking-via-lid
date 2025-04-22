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
from grokking.config_classes.local_estimates.plot_config import LocalEstminatesPlotConfig, PlotSavingConfig
from grokking.config_classes.local_estimates.pointwise_config import LocalEstimatesPointwiseConfig
from grokking.grokk_replica.grokk_model import GrokkModel
from grokking.scripts.collect_hidden_states import collect_hidden_states, preprocess_hidden_states
from grokking.scripts.dataset_for_topological_analysis import (
    DatasetForTopologicalAnalysis,
)
from grokking.scripts.generate_tsne_visualizations import generate_tsne_visualizations
from grokking.scripts.input_and_hidden_states_array import InputAndHiddenStatesArray
from grokking.typing.enums import DeduplicationMode, NNeighborsMode, Verbosity

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
            msg=f"Running topological analysis for {step + 1 = } ...",  # noqa: G004 - low overhead
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
                    msg=f"Running topological analysis for {dataset_for_topological_analysis.split = } ...",  # noqa: G004 - low overhead
                )

            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg="Collecting hidden states ...",
                )

            input_and_hidden_states_array: InputAndHiddenStatesArray = collect_hidden_states(
                model=model,
                topological_analysis_cfg=topological_analysis_cfg,
                dataset_for_topological_analysis=dataset_for_topological_analysis,
                device=device,
                verbosity=verbosity,
                logger=logger,
            )

            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg="Collecting hidden states DONE",
                )

            # # # #
            # Preprocess the hidden states
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg="Preprocessing hidden states ...",
                )

            topo_sampling_seed = topological_analysis_cfg["sampling_seed"]
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
                desc="Iterating over different parameters for local estimates.",
            ):
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

                if verbosity >= Verbosity.NORMAL:
                    logger.info(
                        msg="Preprocessing hidden states DONE",
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
                        f"{local_estimates_config.config_description}": {
                            f"{local_estimates_config.pointwise.config_description}": {
                                "mean": pointwise_results_array_np.mean(),
                                "std": pointwise_results_array_np.std(),
                            },
                            # The global estimate does not depend on the pointwise config.
                            # If available, the global estimate array contains only a single element.
                            "global": global_estimate_array_np[0] if global_estimate_array_np is not None else None,
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

                # TODO: Save the local estimates results to a file

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

                    local_estimates_plot_config = LocalEstminatesPlotConfig(
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
                        local_estimates_config.config_description,
                        local_estimates_config.pointwise.config_description,
                        f"{step+1=}",
                        f"{dataset_for_topological_analysis.split=}",
                    )

                    generate_tsne_visualizations(
                        input_and_hidden_states_array=input_and_hidden_states_array_preprocessed,
                        pointwise_results_array_np=pointwise_results_array_np,
                        local_estimates_plot_config=local_estimates_plot_config,
                        saved_plots_local_estimates_projection_dir_absolute_path=saved_plots_local_estimates_root_dir,
                        verbosity=verbosity,
                        logger=logger,
                    )

                    if verbosity >= Verbosity.NORMAL:
                        logger.info(
                            msg="Creating projection plot DONE",
                        )

            logger.info(
                msg=f"Running topological analysis for {dataset_for_topological_analysis.split = } ...",  # noqa: G004 - low overhead
            )

    if verbosity >= Verbosity.DEBUG:
        logger.debug(
            msg="Setting model to train mode.",
        )
    model.train()

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Running topological analysis for {step + 1 = } DONE",  # noqa: G004 - low overhead
        )
