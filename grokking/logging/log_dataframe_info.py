"""Log information about a pandas DataFrame or an array."""

import logging
import pprint
from io import StringIO

import pandas as pd
from rich.console import Console
from rich.table import Table

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def log_dataframe_info(
    df: pd.DataFrame,
    df_name: str,
    max_log_rows: int = 20,
    *,
    check_for_nan: bool = True,
    logger: logging.Logger = default_logger,
) -> None:
    """Log information about a pandas DataFrame.

    Args:
    ----
        df (pd.DataFrame):
            The DataFrame to log information about.
        df_name (str):
            The name of the DataFrame.
        max_log_rows (int, optional):
            The maximum number of rows to log for the head and tail of the DataFrame.
            Defaults to 20.
        check_for_nan (bool, optional):
            Whether to check for NaN values in the DataFrame.
            Defaults to True.
        logger (logging.Logger, optional):
            The logger to log information to.
            Defaults to logging.getLogger(__name__).

    Returns:
        None

    Side effects:
        Logs information about the DataFrame to the logger.

    """
    logger.info(
        msg=f"{df_name}.shape:\n{df.shape}",  # noqa: G004 - low overhead
    )
    logger.info(
        msg=f"{df_name}.info():\n{df.info()}",  # noqa: G004 - low overhead
    )
    logger.info(
        msg=f"{df_name}.head({max_log_rows}):\n{df.head(max_log_rows).to_string()}",  # noqa: G004 - low overhead
    )
    logger.info(
        msg=f"{df_name}.tail({max_log_rows}):\n{df.tail(max_log_rows).to_string()}",  # noqa: G004 - low overhead
    )

    if check_for_nan:
        # Check if the dataframe contains NaN values
        has_nan = df.isna().any().any()
        logger.info(
            msg=f"has_nan:\n{has_nan}",  # noqa: G004 - low overhead
        )

        if has_nan:
            logger.warning(
                msg=f"{df_name}.isna().sum():\n{df.isna().sum()}",  # noqa: G004 - low overhead
            )
            logger.warning(
                msg="The dataframe contains NaN values. Please make sure that this is intended.",
            )


def rich_table_to_string(
    df: pd.DataFrame,
    max_rows: int = 10,
    max_col_width: int = 30,
) -> str:
    """Convert a pandas DataFrame into a rich-formatted string table.

    Args:
        df: The DataFrame to convert.
        max_rows: Maximum number of rows to show.
        max_col_width: Maximum character width per column (truncate otherwise).

    Returns:
        A string containing the rich-formatted table.

    """
    table = Table(show_header=True, header_style="bold cyan")

    for column in df.columns:
        table.add_column(str(column), style="magenta", max_width=max_col_width)

    # Optionally truncate the DataFrame to avoid huge logs
    display_df = df.head(max_rows)

    for _, row in display_df.iterrows():
        formatted_row = [
            str(val) if len(str(val)) <= max_col_width else str(val)[: max_col_width - 3] + "..." for val in row
        ]
        table.add_row(*formatted_row)

    # Capture the printed output into a string buffer
    console = Console(file=StringIO(), width=100)
    console.print(table)
    return console.file.getvalue()  # type: ignore - typing problem with IO
