import logging
from dataclasses import dataclass

import numpy as np

from grokking.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
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
        sorted_indices_of_original_array: np.ndarray = np.sort(
            a=indices_of_original_array,
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
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg="Returning now without modifying the hidden states and input x.",
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
