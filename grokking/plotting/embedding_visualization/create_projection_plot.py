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


"""Create and save a projection plot from t-SNE results and metadata."""

import logging
import os
import pathlib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio

from grokking.logging.log_dataframe_info import log_dataframe_info
from grokking.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def sanitize_html(
    text: str,
) -> str:
    """Sanitize HTML tags in a string by escaping < and > characters.

    Args:
    ----
        text: The string to sanitize.

    Returns:
    -------
        Sanitized string with < and > characters replaced with &lt; and &gt;.

    """
    return text.replace("<", "&lt;").replace(">", "&gt;")


def create_projection_plot(
    tsne_result: np.ndarray,
    meta_df: pd.DataFrame,
    results_array_np: np.ndarray | None = None,
    maximum_number_of_points: int | None = None,
    text_column_name: str | None = None,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> tuple[
    go.Figure,
    pd.DataFrame,
]:
    """Create a plot annotated with the metadata and save it.

    Args:
    ----
        tsne_result:
            The t-SNE result array, its coordinates are used for the plot.
        meta_df:
            The metadata DataFrame, used for annotating the points in the plot.
        results_array_np:
            Optional, the results array used for coloring the points in the plot.
        verbosity:
            The verbosity level.
        logger:
            The logger object.

    Returns:
    -------
        figure:
            The plotly figure object.
        tsne_df:
            The DataFrame used for the plot, including the t-SNE results and the metadata
            (with truncated elements for better display).

    """
    tsne_df = pd.DataFrame(
        data=tsne_result,
        columns=[
            "TSNE-1",
            "TSNE-2",
        ],
    )
    tsne_df: pd.DataFrame = pd.concat(
        objs=[
            tsne_df,
            meta_df.reset_index(),
        ],
        axis=1,
    )

    # Add a new column for the local estimates, initialize with NaN
    estimate_column_name = "estimate"
    tsne_df[estimate_column_name] = np.nan

    # If results_array_np is provided, populate the estimate column
    if results_array_np is not None:
        tsne_df.loc[
            : len(results_array_np) - 1,
            estimate_column_name,
        ] = results_array_np

    # # # #
    # If a maximum number of points is specified, we only keep the first n points
    if maximum_number_of_points is not None:
        tsne_df = tsne_df.head(
            maximum_number_of_points,
        )

    figure: go.Figure = px.scatter(
        data_frame=tsne_df,
        x="TSNE-1",
        y="TSNE-2",
        text=text_column_name,
        color=estimate_column_name,
        hover_data={
            text_column_name: True,
            estimate_column_name: True,
        },
        color_continuous_scale="Viridis",  # Optional: Choose a color scale
    )

    return figure, tsne_df


def save_projection_plot(  # noqa: PLR0913 - more arguments because of feature flags
    figure: go.Figure,
    tsne_df: pd.DataFrame,
    output_folder: os.PathLike,
    *,
    save_html: bool = True,
    save_pdf: bool = True,
    save_csv: bool = True,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Save the projection plot as HTML and PDF, and the tsne_df as CSV.

    Args:
    ----
        figure:
            The plotly figure object.
        tsne_df:
            The DataFrame used for the plot, including the t-SNE results and the metadata.
        output_folder:
            The output folder for the plot files.
            Will be created if it does not exist, and the plot will be saved as HTML and PDF.
        save_html:
            Whether to save the plot as HTML.
        save_pdf:
            Whether to save the plot as PDF.
        save_csv:
            Whether to save the tsne_df as CSV.
        verbosity:
            The verbosity level.
        logger:
            The logger object.

    """
    pathlib.Path(output_folder).mkdir(
        parents=True,
        exist_ok=True,
    )

    if save_html:
        html_file = pathlib.Path(
            output_folder,
            "tsne_plot.html",
        )
        pio.write_html(
            fig=figure,
            file=html_file,
            auto_open=False,
        )
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Plot saved as HTML to {html_file = }",  # noqa: G004 - low overhead
            )

    if save_pdf:
        pdf_file = pathlib.Path(
            output_folder,
            "tsne_plot.pdf",
        )
        pio.write_image(
            fig=figure,
            file=pdf_file,
            format="pdf",
            width=2400,
            height=1600,
        )
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                f"Plot saved as PDF to {pdf_file = }",  # noqa: G004 - low overhead
            )

    if save_csv:
        csv_file = pathlib.Path(
            output_folder,
            "tsne_df.csv",
        )
        tsne_df.to_csv(
            csv_file,
            index=False,
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                f"tsne_df saved as CSV to {csv_file = }",  # noqa: G004 - low overhead
            )
            log_dataframe_info(
                tsne_df,
                df_name="tsne_df",
                logger=logger,
            )
