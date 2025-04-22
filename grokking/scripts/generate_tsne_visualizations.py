import logging
import pathlib

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from grokking.config_classes.local_estimates.plot_config import LocalEstminatesPlotConfig
from grokking.plotting.embedding_visualization.create_projected_data import create_projected_data
from grokking.plotting.embedding_visualization.create_projection_plot import (
    create_projection_plot,
    save_projection_plot,
)
from grokking.scripts.input_and_hidden_states_array import InputAndHiddenStatesArray
from grokking.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def generate_tsne_visualizations(
    input_and_hidden_states_array: InputAndHiddenStatesArray,
    pointwise_results_array_np: np.ndarray | None,
    local_estimates_plot_config: LocalEstminatesPlotConfig,
    saved_plots_local_estimates_projection_dir_absolute_path: pathlib.Path | None = None,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Generate t-SNE visualizations of the local estimates."""
    tsne_array: np.ndarray = create_projected_data(
        array=input_and_hidden_states_array.hidden_states,
        pca_n_components=local_estimates_plot_config.pca_n_components,
        tsne_n_components=local_estimates_plot_config.tsne_n_components,
        tsne_random_state=local_estimates_plot_config.tsne_random_state,
        verbosity=verbosity,
        logger=logger,
    )

    for maximum_number_of_points in tqdm(
        iterable=[
            1_000,
        ],
        desc="Creating projection plots with different number of points",
    ):
        input_ids_column_name = "input_ids"
        meta_df = pd.DataFrame(
            data=input_and_hidden_states_array.input_x,
            columns=[input_ids_column_name],
        )

        (
            figure,
            tsne_df,
        ) = create_projection_plot(
            tsne_result=tsne_array,
            meta_df=meta_df,
            results_array_np=pointwise_results_array_np,
            maximum_number_of_points=maximum_number_of_points,
            text_column_name=input_ids_column_name,
            verbosity=verbosity,
            logger=logger,
        )

        number_of_points_in_plot: int = len(tsne_df)

        if saved_plots_local_estimates_projection_dir_absolute_path is not None:
            output_folder = pathlib.Path(
                saved_plots_local_estimates_projection_dir_absolute_path,
                f"no-points-in-plot-{number_of_points_in_plot}",
            )
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg=f"Saving projection plot to {output_folder = }",  # noqa: G004 - low overhead
                )

            save_projection_plot(
                figure=figure,
                tsne_df=tsne_df,
                output_folder=output_folder,
                save_html=local_estimates_plot_config.saving.save_html,
                save_pdf=local_estimates_plot_config.saving.save_pdf,
                save_csv=local_estimates_plot_config.saving.save_csv,
                verbosity=verbosity,
                logger=logger,
            )

            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg=f"Saving projection plot to {output_folder = } DONE",  # noqa: G004 - low overhead
                )
