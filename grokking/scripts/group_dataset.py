from typing import Self

import torch
from torch.utils.data import IterableDataset

from grokking.grokk_replica.datasets import AbstractDataset


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
