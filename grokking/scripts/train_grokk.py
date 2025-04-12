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
from dataclasses import dataclass
from typing import Self

import hydra
import hydra.core
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm

import wandb
from grokking.analysis.local_estimates_computation.global_and_pointwise_local_estimates_computation import (
    global_and_pointwise_local_estimates_computation,
)
from grokking.config_classes.local_estimates.plot_config import LocalEstminatesPlotConfig, PlotSavingConfig
from grokking.grokk_replica.datasets import AbstractDataset
from grokking.grokk_replica.grokk_model import GrokkModel
from grokking.grokk_replica.load_objs import load_item
from grokking.grokk_replica.utils import combine_logs
from grokking.logging.create_and_configure_global_logger import create_and_configure_global_logger
from grokking.model_handling.get_torch_device import get_torch_device
from grokking.model_handling.set_seed import set_seed
from grokking.plotting.embedding_visualization.create_projected_data import create_projected_data
from grokking.plotting.embedding_visualization.create_projection_plot import (
    create_projection_plot,
    save_projection_plot,
)
from grokking.typing.enums import Verbosity

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

    train_cfg = config["train"]
    wandb_cfg = config["wandb"]
    topological_analysis_cfg: dict = config["topological_analysis"]

    # Use the global seed to initialize the random number generators and torch initialization.
    global_seed = train_cfg["global_seed"]
    set_seed(
        seed=global_seed,
        logger=logger,
    )

    if wandb_cfg["use_wandb"]:
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

    # # # #
    # Training loop
    for step, (
        x,
        y,
    ) in enumerate(
        iterable=tqdm(train_dataloader),
    ):
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

            if wandb_cfg["use_wandb"]:
                wandb.log(
                    data=out_log,
                )

            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg=f"Running evaluation for {step + 1 = } DONE",  # noqa: G004 - low overhead
                )

        # # # #
        # Embedding space analysis step
        topological_analysis_compute_estimates_every = topological_analysis_cfg["compute_estimates_every"]
        topological_analysis_create_projection_plot_every = topological_analysis_cfg["create_projection_plot_every"]

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
                device=device,
                verbosity=verbosity,
                logger=logger,
            )

        # # # #
        # Finalize training loop step

        # Break condition
        if train_cfg["max_steps"] is not None and step >= train_cfg["max_steps"]:
            break


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
        for i, (val_x, val_y) in tqdm(enumerate(val_dataloader)):
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

                # This list will accumulate the hidden states
            selected_hidden_states_list = []
            selected_input_x_list = []

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

                # # # #
                # Preprocess the hidden states
            topo_number_of_samples = topological_analysis_cfg["number_of_samples"]
            topo_sampling_seed = topological_analysis_cfg["sampling_seed"]

            input_and_hidden_states_array.deduplicate_hidden_states()
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg=f"After deduplication:\n"  # noqa: G004 - low overhead
                    f"{input_and_hidden_states_array!s}",
                )

            input_and_hidden_states_array.subsample(
                number_of_samples=topo_number_of_samples,
                sampling_seed=topo_sampling_seed,
                verbosity=verbosity,
                logger=logger,
            )
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg=f"After subsampling:\n"  # noqa: G004 - low overhead
                    f"{input_and_hidden_states_array!s}",
                )

            # # # #
            # Analyse the extracted hidden states

            # TODO: Add this function call
            #
            # (
            #     global_estimate_array_np,
            #     pointwise_results_array_np,
            # ) = global_and_pointwise_local_estimates_computation(
            #     array_for_estimator=array_for_estimator,
            #     local_estimates_config=main_config.local_estimates,
            #     verbosity=verbosity,
            #     logger=logger,
            # )

            logger.warning(
                msg="@@@ The analysis is not fully implemented yet!",
            )

            # TODO: Implement the analysis here

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
                    f"{step+1=}",
                    f"{dataset_for_topological_analysis.split=}",
                )

                generate_tsne_visualizations(
                    input_and_hidden_states_array=input_and_hidden_states_array,
                    pointwise_results_array_np=None,  # TODO: Replace with the actual results array once implemented
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
