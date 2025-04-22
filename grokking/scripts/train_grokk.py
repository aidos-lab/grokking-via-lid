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

import hydra
import hydra.core
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import wandb
from grokking.config_classes.constants import GROKKING_REPOSITORY_BASE_PATH
from grokking.grokk_replica.datasets import AbstractDataset
from grokking.grokk_replica.grokk_model import GrokkModel
from grokking.grokk_replica.load_objs import load_item
from grokking.grokk_replica.utils import combine_logs
from grokking.logging.create_and_configure_global_logger import create_and_configure_global_logger
from grokking.logging.log_dataframe_info import rich_table_to_string
from grokking.logging.log_model_info import log_model_info
from grokking.model_handling.count_trainable_parameters import count_trainable_parameters
from grokking.model_handling.get_torch_device import get_torch_device
from grokking.model_handling.set_seed import set_seed
from grokking.scripts.dataset_for_topological_analysis import DatasetForTopologicalAnalysis
from grokking.scripts.do_topological_analysis_step import do_topological_analysis_step
from grokking.scripts.group_dataset import GroupDataset
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
default_device: torch.device = torch.device(
    device="cpu",
)


@dataclass
class TrainingLoopState:
    """Class to hold the state of the training loop."""

    model: GrokkModel
    optimizer: torch.optim.Optimizer
    lr_schedule: torch.optim.lr_scheduler.LambdaLR
    training_logs: dict

    dataset: AbstractDataset
    train_dataloader: DataLoader
    val_dataloader: DataLoader
    datasets_for_topological_analysis_list: list[DatasetForTopologicalAnalysis]

    output_dir: os.PathLike

    device: torch.device = default_device
    step: int = 0

    def log_info(
        self,
        logger: logging.Logger = default_logger,
    ) -> None:
        logger.info(
            msg=f"model:\n{self.model}",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"Number of trainable parameters: {count_trainable_parameters(model=self.model) = }",  # noqa: G004 - low overhead
        )
        log_model_info(
            model=self.model,
            model_name="model",
            logger=logger,
        )
        logger.info(
            msg=f"optimizer:\n{self.optimizer}",  # noqa: G004 - low overhead
        )
        # Note: The lr_schedule does not have a useful string representation,
        # so we just print the class name.
        logger.info(
            msg=f"lr_schedule:\n{self.lr_schedule.__class__.__name__}",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"{self.step = }",  # noqa: G004 - low overhead
        )

    @property
    def checkpoints_root_dir(
        self,
    ) -> pathlib.Path:
        """Get the checkpoints root directory."""
        path = pathlib.Path(
            self.output_dir,
            "checkpoints",
            f"step={self.step}",
        )
        return path

    @property
    def checkpoint_data_dict_save_path(
        self,
    ) -> pathlib.Path:
        """Get the checkpoint data dict save path."""
        # Using common torch convention to save checkpoints using the .tar file extension
        path = pathlib.Path(
            self.checkpoints_root_dir,
            "checkpoint_data_dict.tar",
        )
        return path

    @property
    def dataloaders_dict_save_path(
        self,
    ) -> pathlib.Path:
        """Get the dataloaders dict save path."""
        # Using common torch convention to save checkpoints using the .tar file extension
        path = pathlib.Path(
            self.checkpoints_root_dir,
            "dataloaders_dict.tar",
        )
        return path

    def save_to_folder(
        self,
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = default_logger,
    ) -> None:
        """Save the state to a folder."""
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Saving checkpoint for {self.step = } ...",  # noqa: G004 - low overhead
            )

        # # # #
        # Save step 1: Save the model and optimizer state.
        #
        # We will follow the following guide in the torch documentation for saving the model and optimizer state:
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training
        checkpoint_data_dict: dict = {
            "step": self.step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_schedule": self.lr_schedule.state_dict(),
            "training_logs": self.training_logs,
        }

        if not self.checkpoint_data_dict_save_path.exists():
            self.checkpoint_data_dict_save_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Saving checkpoint data dict to {self.checkpoint_data_dict_save_path = } ...",  # noqa: G004 - low overhead
            )
        torch.save(
            obj=checkpoint_data_dict,
            f=self.checkpoint_data_dict_save_path,
        )
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Saving checkpoint data dict to {self.checkpoint_data_dict_save_path = } DONE",  # noqa: G004 - low overhead
            )

        # # # #
        # Save step 2: Save the dataloaders.
        dataloaders_dict: dict = {
            "dataset": self.dataset,
            "train_dataloader": self.train_dataloader,
            "val_dataloader": self.val_dataloader,
            "datasets_for_topological_analysis_list": self.datasets_for_topological_analysis_list,
        }

        if not self.dataloaders_dict_save_path.exists():
            self.dataloaders_dict_save_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Saving dataloaders dict to {self.dataloaders_dict_save_path = } ...",  # noqa: G004 - low overhead
            )
        torch.save(
            obj=dataloaders_dict,
            f=self.dataloaders_dict_save_path,
        )
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Saving dataloaders dict to {self.dataloaders_dict_save_path = } DONE",  # noqa: G004 - low overhead
            )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Saving checkpoint for {self.step = } DONE",  # noqa: G004 - low overhead
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
    use_wandb: bool = wandb_cfg["use_wandb"]
    load_checkpoint_from_dir: None | str = train_cfg["load_checkpoint_from_dir"]

    # Use the global seed to initialize the random number generators and torch initialization.
    global_seed: int = train_cfg["global_seed"]
    set_seed(
        seed=global_seed,
        logger=logger,
    )

    device: torch.device = get_torch_device(
        preferred_torch_backend=train_cfg["preferred_torch_backend"],
        verbosity=verbosity,
        logger=logger,
    )

    if load_checkpoint_from_dir is not None:
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Loading from directory: {load_checkpoint_from_dir = } ...",  # noqa: G004 - low overhead
            )

        # Load the checkpoint and dataloaders from the specified directory.

        # TODO: Implement this
        logger.warning(
            msg="Loading from checkpoint dir is not fully implemented yet.",
        )
        msg = "Loading from checkpoint dir is not fully implemented yet."
        raise NotImplementedError(
            msg,
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Loading from directory: {load_checkpoint_from_dir = } DONE",  # noqa: G004 - low overhead
            )

        # TODO: Make sure that we increase the step by 1 after loading the checkpoint, because this is necessary to be consistent for the next training step iteration.
    else:
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Preparing to train from scratch ...",
            )

        # # # # # # # #
        # Datasets
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

        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=train_cfg["lr"],
            weight_decay=train_cfg["weight_decay"],
            betas=train_cfg["betas"],
        )
        lr_schedule = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda s: min(s / train_cfg["warmup_steps"], 1),
        )

        # Set step to 0 when starting training from scratch.
        step = 0

        training_loop_state: TrainingLoopState = TrainingLoopState(
            model=model,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            training_logs={},
            dataset=dataset,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            datasets_for_topological_analysis_list=datasets_for_topological_analysis_list,
            output_dir=output_dir,
            device=device,
            step=step,
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Preparing to train from scratch DONE",
            )

    if verbosity >= Verbosity.NORMAL:
        training_loop_state.log_info(
            logger=logger,
        )

    training_log_example_batch_every = logging_cfg["training"]["log_example_batch_every"]
    number_of_entries_in_example_batch = logging_cfg["training"]["number_of_entries_in_example_batch"]
    save_checkpoints_every = train_cfg["save_checkpoints_every"]
    topological_analysis_compute_estimates_every = topological_analysis_cfg["compute_estimates_every"]
    topological_analysis_create_projection_plot_every = topological_analysis_cfg["create_projection_plot_every"]

    # # # #
    # Training loop
    training_loop_state.model.train()

    for (
        x,
        y,
    ) in tqdm(
        training_loop_state.train_dataloader,
        desc="Training loop.",
    ):
        if training_log_example_batch_every > 0 and training_loop_state.step % training_log_example_batch_every == 0:
            log_example_batch(
                x=x,
                y=y,
                dataset=training_loop_state.dataset,
                step=training_loop_state.step,
                number_of_entries_in_example_batch=number_of_entries_in_example_batch,
                verbosity=verbosity,
                logger=logger,
            )

        training_logs: dict = do_training_step(
            model=training_loop_state.model,
            optimizer=training_loop_state.optimizer,
            lr_schedule=training_loop_state.lr_schedule,
            x=x,
            y=y,
            device=training_loop_state.device,
        )

        # # # #
        # Evaluation step
        if (training_loop_state.step + 1) % train_cfg["eval_every"] == 0:
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg=f"Running evaluation for {training_loop_state.step + 1 = } ...",  # noqa: G004 - low overhead
                )

            all_val_logs: list[dict] = do_eval_step(
                model=training_loop_state.model,
                val_dataloader=training_loop_state.val_dataloader,
                train_cfg=train_cfg,
                device=training_loop_state.device,
                verbosity=verbosity,
                logger=logger,
            )

            out_log: dict = {
                "val": combine_logs(logs=all_val_logs),
                "train": combine_logs(logs=[training_logs]),
                "step": (training_loop_state.step + 1),
                "lr": float(training_loop_state.lr_schedule.get_last_lr()[0]),
            }
            training_loop_state.training_logs = out_log

            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg=f"{training_loop_state.training_logs}",  # noqa: G004 - low overhead
                )

            if use_wandb:
                # Example of the `training_loop_state.training_logs` dict:
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
                    data=training_loop_state.training_logs,
                )

            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg=f"Running evaluation for {training_loop_state.step + 1 = } DONE",  # noqa: G004 - low overhead
                )

        # # # #
        # Embedding space analysis step

        if (
            topological_analysis_compute_estimates_every > 0
            and (training_loop_state.step + 1) % topological_analysis_compute_estimates_every == 0
        ):
            do_topological_analysis_step(
                datasets_for_topological_analysis_list=training_loop_state.datasets_for_topological_analysis_list,
                model=training_loop_state.model,
                output_dir=training_loop_state.output_dir,
                topological_analysis_cfg=topological_analysis_cfg,
                step=training_loop_state.step,
                topological_analysis_create_projection_plot_every=topological_analysis_create_projection_plot_every,
                use_wandb=use_wandb,
                device=training_loop_state.device,
                verbosity=verbosity,
                logger=logger,
            )

        # # # #
        # Optionally: Save the model, optimizer and dataloader
        #
        # Notes:
        # - We use `step` instead of `step + 1` here,
        #   because we want to also save the model for step == 0,
        #   i.e., at the beginning of training.
        # - We save the dataloaders at the end of the training loop step,
        #   so that we can resume training from the last checkpoint,
        #   and the iterable loaders are at the correct position for the next step.
        if save_checkpoints_every > 0 and training_loop_state.step % save_checkpoints_every == 0:
            training_loop_state.save_to_folder(
                verbosity=verbosity,
                logger=logger,
            )

        # # # #
        # Finalize training loop step
        training_loop_state.step += 1  # noqa: SIM113 - we want to have manual control over the step variable

        # Break condition
        if train_cfg["max_steps"] is not None and training_loop_state.step >= train_cfg["max_steps"]:
            logger.info(
                msg=f"Training loop finished after {training_loop_state.step + 1} steps by reaching max steps.",  # noqa: G004 - low overhead
            )
            break


def log_example_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    dataset: AbstractDataset,
    step: int,
    number_of_entries_in_example_batch: int,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Log an example batch of data."""
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
    optimizer: torch.optim.Optimizer,
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
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
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
        for (
            i,
            (
                val_x,
                val_y,
            ),
        ) in tqdm(
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

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg="Main function ...",
        )

    cfg_as_container = OmegaConf.to_container(
        cfg=cfg,
        resolve=True,
    )
    if not isinstance(
        cfg_as_container,
        dict,
    ):
        msg = "The configuration must be a dictionary."
        raise TypeError(
            msg,
        )
    config: dict = cfg_as_container

    cwd: pathlib.Path = pathlib.Path.cwd()
    hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir  # type: ignore - problem with this import type

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Current working directory:\n{cwd = }",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"Hydra output directory:\n{hydra_output_dir = }",  # noqa: G004 - low overhead
        )

    wandb_cfg: dict = config["wandb"]
    use_wandb: bool = wandb_cfg["use_wandb"]

    if use_wandb:
        logger.info(
            msg=f"Initializing wandb with project name: {wandb_cfg['wandb_project']}",  # noqa: G004 - low overhead
        )

        # Save wandb files to a subdirectory with the project name
        wandb_output_dir = pathlib.Path(
            GROKKING_REPOSITORY_BASE_PATH,
            wandb_cfg["wandb_dir"],
        )
        if not wandb_output_dir.exists():
            logger.info(
                msg=f"Creating wandb output directory: {wandb_output_dir=}",  # noqa: G004 - low overhead
            )
            wandb_output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

        wandb.init(
            project=wandb_cfg["wandb_project"],
            dir=wandb_output_dir,
            notes=wandb_cfg["wandb_notes"],
            config=config,
        )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg="Calling train() function ...",
        )

    train(
        config=config,
        output_dir=hydra_output_dir,
        verbosity=verbosity,
        logger=logger,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg="Calling train() function DONE",
        )

    if use_wandb:
        logger.info(
            msg="Finishing wandb project.",
        )
        wandb.finish()

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg="Main function DONE",
        )


if __name__ == "__main__":
    main()
