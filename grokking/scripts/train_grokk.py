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
import pprint
from typing import Self

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm

import wandb
from grokking.grokk_replica.datasets import AbstractDataset
from grokking.grokk_replica.grokk_model import GrokkModel
from grokking.grokk_replica.load_objs import load_item
from grokking.grokk_replica.utils import combine_logs
from grokking.logging.create_and_configure_global_logger import create_and_configure_global_logger
from grokking.model_handling.get_torch_device import get_torch_device
from grokking.typing.enums import Verbosity

# Increase the wandb service wait time to prevent errors on HHU Hilbert.
# https://github.com/wandb/wandb/issues/5214
os.environ["WANDB__SERVICE_WAIT"] = "300"

# Logger for this file
global_logger: logging.Logger = create_and_configure_global_logger(
    name=__name__,
    file=__file__,
)
default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


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


def train(
    config: dict,
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
    # Dataset and Dataloader

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
    step = 0
    for (
        x,
        y,
    ) in tqdm(train_dataloader):
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

        # # # #
        # Embedding space analysis step
        topological_analysis_compute_estimates_every = topological_analysis_cfg["compute_estimates_every"]

        if topological_analysis_compute_estimates_every < 0:
            logger.info(
                msg="Skipping topological analysis step.",
            )
        elif (step + 1) % topological_analysis_compute_estimates_every == 0:
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg=f"Running topological analysis for {step + 1 = } ...",  # noqa: G004 - low overhead
                )

            logger.warning(
                msg="@@@ This step is not implemented yet!",
            )

            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg=f"Running topological analysis for {step + 1 = } DONE",  # noqa: G004 - low overhead
                )

        # # # #
        # Finalize training loop step
        step += 1

        # Break condition
        if train_cfg["max_steps"] is not None and step >= train_cfg["max_steps"]:
            break


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

    train(
        config=cfg_as_container,
        verbosity=verbosity,
        logger=logger,
    )


if __name__ == "__main__":
    main()
