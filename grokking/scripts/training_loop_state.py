import logging
import os
import pathlib
import pprint
import random
from dataclasses import dataclass
from typing import Self

import numpy as np
import torch
from torch.optim.lr_scheduler import SequentialLR
from torch.utils.data import DataLoader

from grokking.config_classes.constants import default_output_dir
from grokking.grokk_replica.datasets import AbstractDataset
from grokking.logging.log_model_info import log_model_info
from grokking.model_handling.count_trainable_parameters import count_trainable_parameters
from grokking.scripts.dataset_for_topological_analysis import DatasetForTopologicalAnalysis
from grokking.scripts.lr_scheduler_config import LRSchedulerConfig

from grokking.typing.constants import LR_SCHEDULER_CLASS, MODEL_CLASS, OPTIMIZER_CLASS
from grokking.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)
default_device: torch.device = torch.device(
    device="cpu",
)


@dataclass
class TrainingLoopState:
    """Class to hold the state of the training loop."""

    model_init_kwargs: dict
    model: MODEL_CLASS

    optimizer_init_kwargs: dict
    optimizer: torch.optim.Optimizer

    lr_scheduler_init_kwargs: dict
    lr_scheduler: LR_SCHEDULER_CLASS

    training_logs: dict

    dataset: AbstractDataset
    train_dataloader: DataLoader
    val_dataloader: DataLoader
    datasets_for_topological_analysis_list: list[DatasetForTopologicalAnalysis]

    output_dir: os.PathLike

    device: torch.device = default_device
    step: int = 0

    # Using common torch convention to save checkpoints using the .tar file extension
    _CHECKPOINT_DATA_DICT_FILE_NAME: str = "checkpoint_data_dict.tar"
    _DATALOADERS_DICT_FILE_NAME: str = "dataloaders_dict.tar"
    _RNG_STATES_FILE_NAME: str = "rng_states.tar"

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
        # so we just print the class name and the __dict__.
        logger.info(
            msg=f"lr_schedule:\n{self.lr_scheduler.__class__.__name__}",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"lr_schedule.__dict__:\n{pprint.pformat(object=self.lr_scheduler.__dict__)}",  # noqa: G004 - low overhead
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

    @classmethod
    def get_checkpoint_data_dict_save_path(
        cls,
        checkpoints_root_dir: os.PathLike,
    ) -> pathlib.Path:
        """Get the checkpoint data dict save path."""
        path = pathlib.Path(
            checkpoints_root_dir,
            cls._CHECKPOINT_DATA_DICT_FILE_NAME,
        )
        return path

    @classmethod
    def get_dataloaders_dict_save_path(
        cls,
        checkpoints_root_dir: os.PathLike,
    ) -> pathlib.Path:
        """Get the dataloaders dict save path."""
        path = pathlib.Path(
            checkpoints_root_dir,
            cls._DATALOADERS_DICT_FILE_NAME,
        )
        return path

    @classmethod
    def get_rng_states_save_path(
        cls,
        checkpoints_root_dir: os.PathLike,
    ) -> pathlib.Path:
        """Get the RNG state save path."""
        path = pathlib.Path(
            checkpoints_root_dir,
            cls._RNG_STATES_FILE_NAME,
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
            "model_init_kwargs": self.model_init_kwargs,
            "model.state_dict": self.model.state_dict(),
            "optimizer_init_kwargs": self.optimizer_init_kwargs,
            "optimizer.state_dict": self.optimizer.state_dict(),
            "lr_scheduler_init_kwargs": self.lr_scheduler_init_kwargs,
            "lr_scheduler.state_dict": self.lr_scheduler.state_dict(),
            "training_logs": self.training_logs,
        }

        checkpoint_data_dict_save_path: pathlib.Path = self.get_checkpoint_data_dict_save_path(
            checkpoints_root_dir=self.checkpoints_root_dir,
        )
        if not checkpoint_data_dict_save_path.exists():
            checkpoint_data_dict_save_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Saving checkpoint data dict to {checkpoint_data_dict_save_path = } ...",  # noqa: G004 - low overhead
            )
        torch.save(
            obj=checkpoint_data_dict,
            f=checkpoint_data_dict_save_path,
        )
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Saving checkpoint data dict to {checkpoint_data_dict_save_path = } DONE",  # noqa: G004 - low overhead
            )

        # # # #
        # Save step 2: Save the dataloaders.
        dataloaders_dict: dict = {
            "dataset": self.dataset,
            "train_dataloader": self.train_dataloader,
            "val_dataloader": self.val_dataloader,
            "datasets_for_topological_analysis_list": self.datasets_for_topological_analysis_list,
        }

        dataloaders_dict_save_path: pathlib.Path = self.get_dataloaders_dict_save_path(
            checkpoints_root_dir=self.checkpoints_root_dir,
        )
        if not dataloaders_dict_save_path.exists():
            dataloaders_dict_save_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Saving dataloaders dict to {dataloaders_dict_save_path = } ...",  # noqa: G004 - low overhead
            )
        torch.save(
            obj=dataloaders_dict,
            f=dataloaders_dict_save_path,
        )
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Saving dataloaders dict to {dataloaders_dict_save_path = } DONE",  # noqa: G004 - low overhead
            )

        # # # #
        # Save step 3: Save the random states.
        #
        # Notes:
        # - We do not want to put the random state into the checkpoint data dict,
        #   because this might lead to problems when mapping the tensors to another device.
        #   When calling torch.set_rng_state(t) with tensor t not on the CPU, you get:
        #   TypeError: RNG state must be a torch.ByteTensor

        # --- Create dict of random states
        rng_states: dict = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),  # noqa: NPY002 - we specifically want to use the global numpy state here
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
        }

        rng_states_save_path: pathlib.Path = self.get_rng_states_save_path(
            checkpoints_root_dir=self.checkpoints_root_dir,
        )
        if not rng_states_save_path.exists():
            rng_states_save_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Saving RNG states to {rng_states_save_path = } ...",  # noqa: G004 - low overhead
            )
        torch.save(
            obj=rng_states,
            f=rng_states_save_path,
        )
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Saving RNG states to {rng_states_save_path = } DONE",  # noqa: G004 - low overhead
            )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Saving checkpoint for {self.step = } DONE",  # noqa: G004 - low overhead
            )

    # ---------------------------------------------------------------- factory
    @classmethod
    def from_checkpoints_root_dir(
        cls,
        checkpoints_root_dir: os.PathLike,
        output_dir: os.PathLike = default_output_dir,
        *,
        map_location: str | torch.device | None = "cpu",
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = default_logger,
    ) -> Self:
        """Instantiate **directly** from an existing checkpoints folder.

        Args:
            checkpoints_root_dir:
                Path like `/run/checkpoints/step=42`.
            map_location:
                `torch.load` device mapping.

        Raises:
            FileNotFoundError:
                If either expected file is missing.
            ValueError:
                If `checkpoints_root_dir` does not match the step pattern.

        """
        checkpoints_root_dir = pathlib.Path(
            checkpoints_root_dir,
        )

        checkpoint_data_dict_save_path: pathlib.Path = cls.get_checkpoint_data_dict_save_path(
            checkpoints_root_dir=checkpoints_root_dir,
        )
        dataloaders_dict_save_path: pathlib.Path = cls.get_dataloaders_dict_save_path(
            checkpoints_root_dir=checkpoints_root_dir,
        )
        rng_states_save_path: pathlib.Path = cls.get_rng_states_save_path(
            checkpoints_root_dir=checkpoints_root_dir,
        )

        for p in (
            checkpoint_data_dict_save_path,
            dataloaders_dict_save_path,
            rng_states_save_path,
        ):
            if not p.is_file():
                raise FileNotFoundError(
                    p,
                )

        # --- Load the blobs ---
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Loading checkpoint data from {checkpoint_data_dict_save_path = } ...",  # noqa: G004 - low overhead
            )
        checkpoint_data: dict = torch.load(
            f=checkpoint_data_dict_save_path,
            map_location=map_location,
            weights_only=False,
        )
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Loading checkpoint data from {checkpoint_data_dict_save_path = } DONE",  # noqa: G004 - low overhead
            )
            logger.info(
                msg=f"{checkpoint_data.keys() = }",  # noqa: G004 - low overhead
            )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Loading dataloaders from {dataloaders_dict_save_path = } ...",  # noqa: G004 - low overhead
            )
        # Note:
        # - We need to set `weights_only=False` for this loading to work,
        #   because otherwise there is an error with the custom types of the dataloaders.
        dataloaders: dict = torch.load(
            f=dataloaders_dict_save_path,
            map_location=map_location,
            weights_only=False,
        )
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Loading dataloaders from {dataloaders_dict_save_path = } DONE",  # noqa: G004 - low overhead
            )
            logger.info(
                msg=f"{dataloaders.keys() = }",  # noqa: G004 - low overhead
            )

        # --- Load and set the random states ---
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Loading RNG states from {rng_states_save_path = } ...",  # noqa: G004 - low overhead
            )
        # Note:
        # - We need to set `weights_only=False` for this loading to work,
        #   otherwise there is an error with the rng states in the dictionary.
        # - Do not set a map_location here, because if the random states are on the wrong device,
        #   this can lead to an error:
        #   TypeError: RNG state must be a torch.ByteTensor
        rng_states: dict = torch.load(
            f=rng_states_save_path,
            weights_only=False,
        )
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Loading RNG states from {rng_states_save_path = } DONE",  # noqa: G004 - low overhead
            )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Setting random states ...",
            )

        random.setstate(
            state=rng_states["python"],
        )
        np.random.set_state(  # noqa: NPY002 - we specifically want to re-seed the global numpy state here
            state=rng_states["numpy"],
        )
        torch.set_rng_state(
            new_state=rng_states["torch"],
        )
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(
                new_states=rng_states["cuda"],
            )
        if torch.backends.mps.is_available():
            torch.mps.set_rng_state(
                new_state=rng_states["mps"],
            )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Setting random states DONE",
            )

        # --- reconstruct the individual objects ---
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Reconstructing the individual objects ...",
            )

        model = MODEL_CLASS(
            **checkpoint_data["model_init_kwargs"],
        )
        model.load_state_dict(
            state_dict=checkpoint_data["model.state_dict"],
        )
        model: MODEL_CLASS = model.to(
            device=map_location,
        )

        optimizer = OPTIMIZER_CLASS(
            params=model.parameters(),
            **checkpoint_data["optimizer_init_kwargs"],
        )
        optimizer.load_state_dict(
            state_dict=checkpoint_data["optimizer.state_dict"],
        )

        lr_scheduler: SequentialLR = LRSchedulerConfig(
            **checkpoint_data["lr_scheduler_init_kwargs"],
        ).build(
            optimizer=optimizer,
            last_step=checkpoint_data["step"],
        )
        lr_scheduler.load_state_dict(
            state_dict=checkpoint_data["lr_scheduler.state_dict"],
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Reconstructing the individual objects DONE",
            )

        reconstructed_object: TrainingLoopState = cls(
            model_init_kwargs=checkpoint_data["model_init_kwargs"],
            model=model,
            optimizer_init_kwargs=checkpoint_data["optimizer_init_kwargs"],
            optimizer=optimizer,
            lr_scheduler_init_kwargs=checkpoint_data["lr_scheduler_init_kwargs"],
            lr_scheduler=lr_scheduler,
            training_logs=checkpoint_data["training_logs"],
            dataset=dataloaders["dataset"],
            train_dataloader=dataloaders["train_dataloader"],
            val_dataloader=dataloaders["val_dataloader"],
            datasets_for_topological_analysis_list=dataloaders["datasets_for_topological_analysis_list"],
            output_dir=output_dir,
            device=torch.device(device=map_location) if map_location is not None else default_device,
            step=checkpoint_data["step"],
        )

        return reconstructed_object
