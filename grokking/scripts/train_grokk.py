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

"""Training script for Grokking models."""

import logging
import os
import pathlib
import pprint
from copy import deepcopy
from dataclasses import dataclass
from io import StringIO
from itertools import product
from typing import Self

import hydra
import hydra.core
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from rich.console import Console, ConsoleOptions, RenderResult
from rich.table import Table
from rich.text import Text
from torch.utils.data import DataLoader, IterableDataset
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
from grokking.grokk_replica.datasets import AbstractDataset
from grokking.grokk_replica.grokk_model import GrokkModel
from grokking.grokk_replica.load_objs import load_item
from grokking.grokk_replica.utils import combine_logs
from grokking.logging.create_and_configure_global_logger import create_and_configure_global_logger
from grokking.logging.log_model_info import log_model_info
from grokking.model_handling.count_trainable_parameters import count_trainable_parameters
from grokking.model_handling.get_torch_device import get_torch_device
from grokking.model_handling.set_seed import set_seed
from grokking.plotting.embedding_visualization.create_projected_data import create_projected_data
from grokking.plotting.embedding_visualization.create_projection_plot import (
    create_projection_plot,
    save_projection_plot,
)
from grokking.typing.enums import DeduplicationMode, NNeighborsMode, Verbosity

# Increase the wandb service wait time to prevent errors on HHU Hilbert.
# https://github.com/wandb/wandb/issues/5214
os.environ["WANDB__SERVICE_WAIT"] = "300"

default_output_dir = pathlib.Path(
    "outputs",
)

# Logger for this file
global_logger: logging.Logger = create_and_configure_global_logger(
    name=__name__,
    file=__file__,
)
default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)
default_device: torch.device = torch.device(device="cpu")


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
    return console.file.getvalue()


class GroupDataset(IterableDataset):
    """Dataset wrapper for training and validation datasets for a given group with operation."""

    def __init__(
        self,
        dataset: AbstractDataset,
        split: str,
    ) -> None:
        """Initialize GroupDataset with dataset and split."""
        super().__init__()

        if split not in {
            "train",
            "val",
        }:
            msg: str = f"Invalid split: {split = }. Must be 'train' or 'val'."
            raise ValueError(
                msg,
            )

        self.dataset = dataset
        self.split: str = split

        if self.split == "train":
            self.fetch_f = self.dataset.fetch_train_example
        elif self.split == "val":
            self.fetch_f = self.dataset.fetch_val_example
        else:
            raise NotImplementedError

    def __iter__(
        self,
    ) -> Self:
        return self

    def __next__(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        # Example outputs of the fetch function for mod_sum_dataset dataset with p=96:
        # > ([29, 0, 32, 1], 57, [27, 'o', 30, '=', 57])
        # > ([27, 0, 37, 1], 60, [25, 'o', 35, '=', 60])
        # > ([92, 0, 40, 1], 32, [90, 'o', 38, '=', 32])
        # Note that the input_ids for the operands are in the range [2, p + 1],
        # since the operator 'o' corresponds to input_id 0 and the equality '=' to input_id 1.
        # The values of y are in the range [0, p - 1],
        # since the model output is predicting only in the operand range,
        # also compare with the 96-dimensional output layer:
        # > (output): Linear(in_features=128, out_features=96, bias=True)
        (
            x,
            y,
            _,
        ) = self.fetch_f()
        return (
            torch.tensor(x),
            torch.tensor(y),
        )


@dataclass
class DatasetForTopologicalAnalysis:
    """Dataset wrapper for topological analysis."""

    dataset: GroupDataset
    split: str
    dataloader: DataLoader

    def __init__(
        self,
        group_dataset: GroupDataset,
        split: str,
        train_cfg: dict,
    ) -> None:
        """Initialize DatasetForTopologicalAnalysis with dataset, split, and dataloader."""
        self.dataset = group_dataset
        self.split = split

        self.dataloader = DataLoader(
            dataset=group_dataset,
            batch_size=train_cfg["bsize"],
            shuffle=False,
            num_workers=train_cfg["num_workers"],
        )


@dataclass
class InputAndHiddenStatesArray:
    """Container for input and hidden states."""

    input_x: list
    hidden_states: np.ndarray

    def __str__(self) -> str:
        return f"InputAndHiddenStatesArray({len(self.input_x)=}; {self.hidden_states.shape=})"

    def deduplicate_hidden_states(
        self,
    ) -> None:
        (
            unique_vectors,
            indices_of_original_array,
        ) = np.unique(
            ar=self.hidden_states,
            axis=0,
            return_index=True,
        )

        # Keep same order of original vectors by sorting the indices
        sorted_indices_of_original_array = np.sort(
            indices_of_original_array,
        )

        # Update the hidden states and input x
        self.hidden_states = self.hidden_states[sorted_indices_of_original_array]
        self.input_x = [self.input_x[i] for i in sorted_indices_of_original_array]

    def subsample(
        self,
        number_of_samples: int,
        sampling_seed: int,
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = default_logger,
    ) -> None:
        """Subsample the hidden states and input x."""
        if number_of_samples > len(self.input_x):
            logger.warning(
                msg="Requested number of samples exceeds available input_x length. We will use all available samples.",
            )
            logger.info(
                msg="We will not modify the hidden states and input x.",
            )
            return

        rng = np.random.default_rng(seed=sampling_seed)
        indices_to_keep = rng.choice(
            a=len(self.input_x),
            size=number_of_samples,
            replace=False,
        )

        # Update the hidden states and input x
        self.hidden_states = self.hidden_states[indices_to_keep]
        self.input_x = [self.input_x[i] for i in indices_to_keep]


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


def train(
    config: dict,
    output_dir: os.PathLike = default_output_dir,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Train the model using the provided configuration."""
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Using config:\n{pprint.pformat(config)}",  # noqa: G004 - low overhead
        )

    train_cfg: dict = config["train"]
    logging_cfg: dict = config["logging"]
    topological_analysis_cfg: dict = config["topological_analysis"]
    wandb_cfg: dict = config["wandb"]

    # Use the global seed to initialize the random number generators and torch initialization.
    global_seed = train_cfg["global_seed"]
    set_seed(
        seed=global_seed,
        logger=logger,
    )

    use_wandb: bool = wandb_cfg["use_wandb"]

    if use_wandb:
        logger.info(
            msg=f"Initializing wandb with project name: {wandb_cfg['wandb_project']}",  # noqa: G004 - low overhead
        )
        wandb.init(
            project=wandb_cfg["wandb_project"],
            config=config,
        )

    device: torch.device = get_torch_device(
        preferred_torch_backend=train_cfg["preferred_torch_backend"],
        verbosity=verbosity,
        logger=logger,
    )

    # # # # # # # #
    # Datasets and Dataloaders

    dataset: AbstractDataset = load_item(
        config["dataset"],
    )
    train_data = GroupDataset(
        dataset=dataset,
        split="train",
    )
    val_data = GroupDataset(
        dataset=dataset,
        split="val",
    )

    # Notes:
    # - We create new instances of the datasets for the topological analysis,
    #   so that we can consume new batches of data for each analysis step.
    #   This is important because the datasets are iterable, so we need to
    #   create new instances to get new batches.
    datasets_for_topological_analysis_list: list[DatasetForTopologicalAnalysis] = [
        DatasetForTopologicalAnalysis(
            group_dataset=GroupDataset(
                dataset=dataset,
                split=split,
            ),
            split=split,
            train_cfg=train_cfg,
        )
        for split in [
            "train",
            "val",
        ]
    ]

    # Notes:
    # - In the current setup without shuffling, the dataloaders will not introduce any non-determinism.
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=train_cfg["bsize"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
    )
    val_dataloader = DataLoader(
        dataset=val_data,
        batch_size=train_cfg["bsize"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
    )

    # # # #
    # Model

    model: GrokkModel = load_item(
        config["model"],
        dataset.n_vocab,
        dataset.n_out,
        device,
    )
    model.train()

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"model:\n{model}",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"Number of trainable parameters: {count_trainable_parameters(model) = }",  # noqa: G004 - low overhead
        )
        log_model_info(
            model=model,
            model_name="model",
            logger=logger,
        )

    optim = torch.optim.AdamW(
        params=model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
        betas=train_cfg["betas"],
    )
    lr_schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optim,
        lr_lambda=lambda s: min(s / train_cfg["warmup_steps"], 1),
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"optimizer:\n{optim}",  # noqa: G004 - low overhead
        )
        # Note: The lr_schedule does not have a useful string representation,
        # so we just print the class name.
        logger.info(
            msg=f"lr_schedule:\n{lr_schedule.__class__.__name__}",  # noqa: G004 - low overhead
        )

    training_log_example_batch_every = logging_cfg["training"]["log_example_batch_every"]
    number_of_entries_in_example_batch = logging_cfg["training"]["number_of_entries_in_example_batch"]
    save_checkpoints_every = train_cfg["save_checkpoints_every"]
    topological_analysis_compute_estimates_every = topological_analysis_cfg["compute_estimates_every"]
    topological_analysis_create_projection_plot_every = topological_analysis_cfg["create_projection_plot_every"]

    # # # #
    # Training loop
    for step, (
        x,
        y,
    ) in enumerate(
        iterable=tqdm(
            train_dataloader,
            desc="Training loop.",
        ),
    ):
        if training_log_example_batch_every > 0 and step % training_log_example_batch_every == 0:
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg=f"Logging example batch for {step = } ...",  # noqa: G004 - low overhead
                )
                logger.info(
                    msg=f"{x.shape = }; {y.shape = }",  # noqa: G004 - low overhead
                )

            # Comment about decoding:
            # `train_dataloader.dataset` is a GroupDataset, we need to go one level deeper to access the instance
            # which inherits from AbstractDataset and provides the decode method.
            # > train_dataloader.dataset.dataset.decode(x[0])
            # Here we still have access to the original dataset, so we can just use this for decoding.

            number_of_entries_to_log: int = min(
                number_of_entries_in_example_batch,
                len(x),
            )
            collected_examples_list: list = [
                batch_to_table_entry(
                    x=x,
                    y=y,
                    index=i,
                    dataset=dataset,
                )
                for i in range(number_of_entries_to_log)
            ]
            collected_examples_df: pd.DataFrame = pd.DataFrame(
                data=collected_examples_list,
            )

            table_str: str = rich_table_to_string(
                df=collected_examples_df,
                max_rows=number_of_entries_in_example_batch,
            )
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg=f"Example batch for {step = }:\n{table_str}",  # noqa: G004 - low overhead
                )

        # # # #
        # Optionally: Save the model, optimizer and dataloader

        if save_checkpoints_every > 0 and step % save_checkpoints_every == 0:
            # Note: We use `step` instead of `step + 1` here,
            # because we want to also save the model for step == 0, i.e., at the beginning of training.
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg=f"Saving checkpoint for {step = } ...",  # noqa: G004 - low overhead
                )

            # TODO: Implement the model saving
            logger.warning(
                msg="Saving is not fully implemented yet!",
            )

            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg=f"Saving checkpoint for {step = } DONE",  # noqa: G004 - low overhead
                )

        training_logs: dict = do_training_step(
            model=model,
            optim=optim,
            lr_schedule=lr_schedule,
            x=x,
            y=y,
            device=device,
        )

        # # # #
        # Evaluation step
        if (step + 1) % train_cfg["eval_every"] == 0:
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg=f"Running evaluation for {step + 1 = } ...",  # noqa: G004 - low overhead
                )

            all_val_logs: list[dict] = do_eval_step(
                model=model,
                val_dataloader=val_dataloader,
                train_cfg=train_cfg,
                device=device,
                verbosity=verbosity,
                logger=logger,
            )

            out_log: dict = {
                "val": combine_logs(logs=all_val_logs),
                "train": combine_logs(logs=[training_logs]),
                "step": (step + 1),
                "lr": float(lr_schedule.get_last_lr()[0]),
            }

            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg=f"{out_log}",  # noqa: G004 - low overhead
                )

            if use_wandb:
                # Example of the `out_log` dict:
                #
                # > {
                # >     "val": {
                # >         "loss": 4.6041805148124695,
                # >         "accuracy": 0.00927734375,
                # >         "attn_entropy": 0.7255031652748585,
                # >         "param_norm": 121.61177041725013,
                # >     },
                # >     "train": {
                # >         "loss": 4.638635158538818,
                # >         "accuracy": 0.015625,
                # >         "attn_entropy": 0.6404158473014832,
                # >         "param_norm": 121.60770918646537,
                # >     },
                # >     "step": 10,
                # >     "lr": 0.001,
                # > }
                wandb.log(
                    data=out_log,
                )

            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg=f"Running evaluation for {step + 1 = } DONE",  # noqa: G004 - low overhead
                )

        # # # #
        # Embedding space analysis step

        if (
            topological_analysis_compute_estimates_every > 0
            and (step + 1) % topological_analysis_compute_estimates_every == 0
        ):
            do_topological_analysis_step(
                datasets_for_topological_analysis_list=datasets_for_topological_analysis_list,
                model=model,
                output_dir=output_dir,
                topological_analysis_cfg=topological_analysis_cfg,
                step=step,
                topological_analysis_create_projection_plot_every=topological_analysis_create_projection_plot_every,
                use_wandb=use_wandb,
                device=device,
                verbosity=verbosity,
                logger=logger,
            )

        # # # #
        # Finalize training loop step

        # Break condition
        if train_cfg["max_steps"] is not None and step >= train_cfg["max_steps"]:
            break


def batch_to_table_entry(
    x: torch.Tensor,
    y: torch.Tensor,
    index: int,
    dataset: AbstractDataset,
) -> dict:
    """Convert a batch to a table entry."""
    entry: dict = {
        "x": x[index].cpu().numpy().tolist(),
        "x_decoded": dataset.decode(sequence=x[index]),
        "y": y[index].cpu().numpy().tolist(),
    }
    return entry


def do_training_step(
    model: GrokkModel,
    optim: torch.optim.Optimizer,
    lr_schedule: torch.optim.lr_scheduler.LambdaLR,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> dict:
    (
        loss,
        logs,
    ) = model.get_loss(
        x=x.to(device),
        y=y.to(device),
    )
    optim.zero_grad()
    loss.backward()
    optim.step()
    lr_schedule.step()

    return logs


def do_eval_step(
    model: GrokkModel,
    val_dataloader: DataLoader,
    train_cfg: dict,
    device: torch.device,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> list[dict]:
    model.eval()

    with torch.no_grad():
        all_val_logs = []
        for i, (val_x, val_y) in tqdm(
            enumerate(iterable=val_dataloader),
            desc="Evaluating validation data.",
        ):
            if i >= train_cfg["eval_batches"]:
                break
            (
                _,
                val_logs,
            ) = model.get_loss(
                val_x.to(device),
                val_y.to(device),
            )
            all_val_logs.append(val_logs)

    model.train()

    return all_val_logs


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

                additional_pointwise_results_statistics: dict = create_additional_pointwise_results_statistics(
                    pointwise_results_array_np=pointwise_results_array_np,
                    truncation_size_range=range(500, 5_000, 500),
                    verbosity=verbosity,
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

    model.train()

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Running topological analysis for {step + 1 = } DONE",  # noqa: G004 - low overhead
        )


def preprocess_hidden_states(
    input_and_hidden_states_array: InputAndHiddenStatesArray,
    topo_sampling_seed: int,
    local_estimates_config: LocalEstimatesConfig,
    verbosity: Verbosity,
    logger: logging.Logger,
) -> InputAndHiddenStatesArray:
    temp_input_and_hidden_states_array: InputAndHiddenStatesArray = deepcopy(
        x=input_and_hidden_states_array,
    )

    temp_input_and_hidden_states_array.deduplicate_hidden_states()
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"After deduplication:\n"  # noqa: G004 - low overhead
            f"{temp_input_and_hidden_states_array!s}",
        )

    temp_input_and_hidden_states_array.subsample(
        number_of_samples=local_estimates_config.filtering.num_samples,
        sampling_seed=topo_sampling_seed,
        verbosity=verbosity,
        logger=logger,
    )
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"After subsampling:\n"  # noqa: G004 - low overhead
            f"{temp_input_and_hidden_states_array!s}",
        )

    return temp_input_and_hidden_states_array


def collect_hidden_states(
    model: torch.nn.Module,
    topological_analysis_cfg: dict,
    dataset_for_topological_analysis: DatasetForTopologicalAnalysis,
    device: torch.device = default_device,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> InputAndHiddenStatesArray:
    # This list will accumulate the hidden states
    selected_hidden_states_list: list = []
    selected_input_x_list: list = []

    # Note:
    # - Currently, we use the same dataloader in each iteration where this topological analysis is run.
    # - Thus, the embedded data is different for each iteration,
    #   since we keep stepping through the iterable dataset.
    for topo_batch_index, (
        topo_input_x,
        topo_input_y,
    ) in enumerate(
        iterable=tqdm(
            dataset_for_topological_analysis.dataloader,
            desc=f"Collecting hidden states for {dataset_for_topological_analysis.split = }",
        ),
    ):
        # Break condition is necessary,
        # because otherwise we would keep looping over the dataset and never stop
        if topo_batch_index >= topological_analysis_cfg["max_number_of_topo_batches"]:
            break

        (
            _,
            _,
            hidden_states_over_layers_list,
        ) = model.forward(
            x=topo_input_x.to(device),
        )

        # Take the hidden states of the last layer.
        # > hidden_states_single_layer.shape = torch.Size([512, 4, 128])
        hidden_states_single_layer = hidden_states_over_layers_list[-1]

        # Currently, we are only interested in the hidden states of the operands,
        # i.e., we want to exclude the operation token "o" and the equality token "=":
        # Only select the 0th and 2nd token embeddings in the batch.
        # > only_operand_hidden_states.shape = torch.Size([512, 2, 128])
        only_operand_hidden_states = hidden_states_single_layer[
            :,
            [0, 2],
            :,
        ]

        # Move the hidden states to the CPU and convert them to a numpy array
        only_operand_hidden_states_np = only_operand_hidden_states.detach().cpu().numpy()
        # Make this into a list of all the 128-dimensional hidden states:
        # I.e., convert the shape from (512, 2, 128) to (512 * 2, 128)
        only_operand_hidden_states_reshaped_np = only_operand_hidden_states_np.reshape(
            -1,
            only_operand_hidden_states_np.shape[-1],
        )
        # Turn this into a list of 128-dimensional hidden states:
        only_operand_hidden_states_list = only_operand_hidden_states_reshaped_np.tolist()
        # Extend the list of hidden states with the new hidden states:
        selected_hidden_states_list.extend(only_operand_hidden_states_list)

        corresponding_input_x_np = (
            topo_input_x[
                :,
                [0, 2],
            ]
            .detach()
            .cpu()
            .numpy()
        ).reshape(
            -1,
        )
        # Extend the list of input x with the new input x:
        selected_input_x_list.extend(corresponding_input_x_np.tolist())

    # Create wrapper object
    input_and_hidden_states_array = InputAndHiddenStatesArray(
        input_x=selected_input_x_list,
        hidden_states=np.array(selected_hidden_states_list),
    )
    if verbosity >= Verbosity.NORMAL:
        # The string representation of the object will print the shapes of the list and array.
        logger.info(
            msg=f"Extracted hidden states container:\n{input_and_hidden_states_array!s}",  # noqa: G004 - low overhead
        )

    return input_and_hidden_states_array


@hydra.main(
    config_path="../../config",
    config_name="train_grokk",
    version_base="1.3",
)
def main(
    cfg: DictConfig,
) -> None:
    """Train the model."""
    logger: logging.Logger = global_logger
    verbosity: Verbosity = Verbosity.NORMAL

    cfg_as_container = OmegaConf.to_container(
        cfg=cfg,
    )
    if not isinstance(
        cfg_as_container,
        dict,
    ):
        msg = "The configuration must be a dictionary."
        raise TypeError(
            msg,
        )

    cwd: pathlib.Path = pathlib.Path.cwd()
    hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir  # type: ignore - problem with this import type

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Current working directory:\n{cwd = }",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"Hydra output directory:\n{hydra_output_dir = }",  # noqa: G004 - low overhead
        )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg="Calling train() function ...",
        )

    train(
        config=cfg_as_container,
        output_dir=hydra_output_dir,
        verbosity=verbosity,
        logger=logger,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg="Calling train() function DONE",
        )


if __name__ == "__main__":
    main()
