"""Datasets for the group operation task."""

import abc
import random
from itertools import permutations
from typing import Any


class AbstractDataset(abc.ABC):
    """Abstract class for datasets."""

    def __init__(
        self,
        group_elements1: set,
        group_elements2: set,
        frac_train: float,
        dataset_seed: int,
    ) -> None:
        self.frac_train: float = frac_train

        self.dataset_seed: int = dataset_seed
        # Create a generator for the random number generator
        self.rng = random.Random(  # noqa: S311 - we will not use this for cryptography
            x=self.dataset_seed,
        )

        self.group_elements1: set = group_elements1
        self.group_elements2: set = group_elements2
        self.ordered_group_elements1 = list(self.group_elements1)
        self.ordered_group_elements2 = list(self.group_elements2)

        # This is the mapping from token_ids to the vocabulary
        # (operation sign, equals sign, and the group elements)
        self.idx2vocab: list = [
            "o",
            "=",
            *list(group_elements1.union(group_elements2)),
        ]
        self.vocab2idx: dict = {vocab: idx for idx, vocab in enumerate(self.idx2vocab)}
        self.n_vocab: int = len(self.idx2vocab)
        # n_out is used to set the output size of the transformer model
        self.n_out: int = len(group_elements1.union(group_elements2))

        # idxs is a list of length of all possible pairs of group elements
        idxs = list(range(len(self.group_elements1) * len(self.group_elements2)))

        # Shuffle the idxs list to create a random order
        self.rng.shuffle(
            x=idxs,
        )

        (
            self.train_pairs,
            self.val_pairs,
        ) = (
            idxs[: int(len(idxs) * frac_train)],
            idxs[int(len(idxs) * frac_train) :],
        )

    @abc.abstractmethod
    def fetch_output(
        self,
        a,
        b,
    ) -> Any:
        pass

    def encode(
        self,
        sequence,
    ) -> list:
        return [self.vocab2idx[item] for item in sequence]

    def decode(
        self,
        sequence,  # Note: 'list' type is too strict here, since we also want to apply this to tensors
    ) -> list:
        return [self.idx2vocab[item] for item in sequence]

    def form_equation(
        self,
        a,
        b,
        c,
    ) -> list:
        return [
            a,
            "o",
            b,
            "=",
            c,
        ]

    def fetch_example(
        self,
        idx: int,
    ) -> tuple[list[int], int, list]:
        """Fetch an example from the dataset."""
        a = self.ordered_group_elements1[idx // len(self.group_elements2)]
        b = self.ordered_group_elements2[idx % len(self.group_elements2)]
        c = self.fetch_output(
            a=a,
            b=b,
        )
        equation: list = self.form_equation(
            a=a,
            b=b,
            c=c,
        )
        # The entries in the returned tuple are:
        # - The encoded equation without the last token (which is the output) --> x in the training batches
        # - The value corresponding to the output --> y in the training batches.
        #   Note:
        #   The output layer has dimension n_out (which is the size of the group) and the first two tokens are "o" and "=".
        #   This is why for the entry at index 1 in the returned tuple,
        #   which corresponds to the y value in the dataset entry, we subtract 2 from the index.
        # - The equation itself (for debugging purposes) --> not used in the training batches.
        return (
            self.encode(sequence=equation[:-1]),
            (self.vocab2idx[c] - 2),
            equation,
        )

    def fetch_train_example(
        self,
    ) -> tuple[list[int], int, list]:
        idx: int = self.rng.choice(
            seq=self.train_pairs,
        )
        return self.fetch_example(
            idx=idx,
        )

    def fetch_val_example(
        self,
    ) -> tuple[list[int], int, list]:
        idx: int = self.rng.choice(
            seq=self.val_pairs,
        )
        return self.fetch_example(
            idx=idx,
        )


class ModSumDataset(AbstractDataset):
    def __init__(
        self,
        p: int,
        frac_train: float,
        dataset_seed: int,
    ) -> None:
        super().__init__(
            group_elements1=set(range(p)),
            group_elements2=set(range(p)),
            frac_train=frac_train,
            dataset_seed=dataset_seed,
        )
        self.p = p

    def fetch_output(
        self,
        a: int,
        b: int,
    ) -> int:
        return (a + b) % self.p


class ModSubtractDataset(AbstractDataset):
    def __init__(
        self,
        p: int,
        frac_train: float,
        dataset_seed: int,
    ) -> None:
        super().__init__(
            group_elements1=set(range(p)),
            group_elements2=set(range(p)),
            frac_train=frac_train,
            dataset_seed=dataset_seed,
        )
        self.p = p

    def fetch_output(
        self,
        a: int,
        b: int,
    ) -> int:
        return (a - b) % self.p


class ModMultiplyDataset(AbstractDataset):
    def __init__(
        self,
        p: int,
        frac_train: float,
        dataset_seed: int,
    ) -> None:
        super().__init__(
            group_elements1=set(range(p)),
            group_elements2=set(range(p)),
            frac_train=frac_train,
            dataset_seed=dataset_seed,
        )
        self.p: int = p

    def fetch_output(
        self,
        a: int,
        b: int,
    ) -> int:
        return (a * b) % self.p


class ModDivisonDataset(AbstractDataset):
    def __init__(
        self,
        p: int,
        frac_train: float,
        dataset_seed: int,
    ) -> None:
        super().__init__(
            group_elements1=set(range(p)),
            group_elements2=set(range(1, p)),
            frac_train=frac_train,
            dataset_seed=dataset_seed,
        )
        self.p = p

    def fetch_output(
        self,
        a: int,
        b: int,
    ) -> int:
        return (a * pow(b, self.p - 2, self.p)) % self.p


class PermutationGroup(AbstractDataset):
    def __init__(
        self,
        k: int,
        frac_train: float,
        dataset_seed: int,
    ) -> None:
        perms = set(
            map(
                tuple,
                permutations(
                    list(range(k)),
                ),
            ),
        )
        super().__init__(
            group_elements1=perms,
            group_elements2=perms,
            frac_train=frac_train,
            dataset_seed=dataset_seed,
        )
        self.k = k

    def fetch_output(  # type: ignore - NOTE: There is a typing problem here, but we need to try out this group before we can debug it.
        self,
        a,
        b,
    ):
        return tuple([a[b[i]] for i in range(len(b))])
