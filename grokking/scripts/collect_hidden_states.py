import logging
from copy import deepcopy

import numpy as np
import torch
from tqdm.auto import tqdm

from grokking.config_classes.local_estimates.local_estimates_config import LocalEstimatesConfig
from grokking.scripts.dataset_for_topological_analysis import (
    DatasetForTopologicalAnalysis,
)
from grokking.scripts.input_and_hidden_states_array import InputAndHiddenStatesArray
from grokking.typing.enums import TokenRestrictionMode, Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)
default_device: torch.device = torch.device(
    device="cpu",
)


def collect_hidden_states(
    model: torch.nn.Module,
    topological_analysis_cfg: dict,
    dataset_for_topological_analysis: DatasetForTopologicalAnalysis,
    token_restriction_mode: TokenRestrictionMode,
    *,
    make_copy_of_dataloader: bool = True,
    device: torch.device = default_device,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> InputAndHiddenStatesArray:
    """Collect hidden states from the model for the given dataset."""
    # This list will accumulate the hidden states
    selected_hidden_states_list: list = []
    selected_input_x_list: list = []

    match make_copy_of_dataloader:
        case True:
            # Make a deepcopy of the dataloader,
            # so that we can iterate over it with the same state multiple times
            dataloader_to_iterate_over = deepcopy(
                x=dataset_for_topological_analysis.dataloader,
            )
        case False:
            # - Here, we re-use the dataloader in each iteration where this topological analysis is run.
            # - Thus, the embedded data is different for each iteration,
            #   since we keep stepping through the iterable dataset.
            dataloader_to_iterate_over = dataset_for_topological_analysis.dataloader

    match token_restriction_mode:
        case TokenRestrictionMode.RESTRICT_TO_OPERANDS:
            # In this case are only interested in the hidden states of the operands,
            # i.e., we want to exclude the operation token "o" and the equality token "=":
            # Only select the 0th and 2nd token embeddings in the batch.
            token_indices_to_select: list[int] = [0, 2]
        case TokenRestrictionMode.TAKE_ALL:
            # In this case we want to take all the tokens,
            # i.e., we want to include the operation token "o" and the equality token "=":
            # Select all the token embeddings in the batch.
            token_indices_to_select: list[int] = [0, 1, 2, 3]
        case _:
            raise ValueError(
                msg=f"Unknown token restriction mode: {token_restriction_mode!s}",
            )

    for topo_batch_index, (
        topo_input_x,
        topo_input_y,
    ) in enumerate(
        iterable=tqdm(
            dataloader_to_iterate_over,
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
        ) = model(
            x=topo_input_x.to(device),
        )

        # Take the hidden states of the last layer.
        # > hidden_states_single_layer.shape = torch.Size([512, 4, 128])
        hidden_states_single_layer = hidden_states_over_layers_list[-1]

        # > only_operand_hidden_states.shape = torch.Size([512, 2 or 4, 128])
        only_restricted_tokens_hidden_states = hidden_states_single_layer[
            :,
            token_indices_to_select,
            :,
        ]

        # Move the hidden states to the CPU and convert them to a numpy array
        only_restricted_tokens_hidden_states_np = only_restricted_tokens_hidden_states.detach().cpu().numpy()
        # Make this into a list of all the 128-dimensional hidden states:
        # I.e., convert the shape from (512, 2 or 4, 128) to (512 * (2 or 4), 128)
        only_restricted_tokens_hidden_states_reshaped_np = only_restricted_tokens_hidden_states_np.reshape(
            -1,
            only_restricted_tokens_hidden_states_np.shape[-1],
        )
        # Turn this into a list of 128-dimensional hidden states:
        only_restricted_tokens_hidden_states_list = only_restricted_tokens_hidden_states_reshaped_np.tolist()
        # Extend the list of hidden states with the new hidden states:
        selected_hidden_states_list.extend(only_restricted_tokens_hidden_states_list)

        corresponding_input_x_np = (
            topo_input_x[
                :,
                token_indices_to_select,
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


def preprocess_hidden_states(
    input_and_hidden_states_array: InputAndHiddenStatesArray,
    topo_sampling_seed: int,
    local_estimates_config: LocalEstimatesConfig,
    verbosity: Verbosity,
    logger: logging.Logger,
) -> InputAndHiddenStatesArray:
    """Preprocess the hidden states."""
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
