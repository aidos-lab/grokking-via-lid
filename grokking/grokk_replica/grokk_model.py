"""Define the Grokking model."""

import torch
import torch.nn.functional as F
from torch import nn

from grokking.grokk_replica.transformer import Transformer
from grokking.grokk_replica.utils import causal_attn_mask, parameter_norm


class GrokkModel(nn.Module):
    def __init__(
        self,
        transformer_config: dict,
        vocab_size: int,
        output_size: int,
        device: torch.device,
    ) -> None:
        """Initialize the GrokkModel with a transformer."""
        super().__init__()
        self.transformer = Transformer(
            **transformer_config,
            vocab_size=vocab_size,
            output_size=output_size,
        )  # type: ignore - The transformer_config contains the necessary arguments for the Transformer class
        self.device = device

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        attn_mask = (
            causal_attn_mask(x.shape[1])
            .unsqueeze(0)
            .repeat(x.shape[0], 1, 1)
            .to(
                self.device,
            )
        )
        (
            predictions,
            attentions_over_layers_list,
            _,
            hidden_states_over_layers_list,
        ) = self.transformer.forward(
            x=x,
            attn_mask=attn_mask,
            past_kvs=None,
        )
        return (
            predictions,
            attentions_over_layers_list,
            hidden_states_over_layers_list,
        )

    def get_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        dict,
    ]:
        # Use model() instead of model.forward() to avoid problems with wandb logging of parameters:
        # https://community.wandb.ai/t/wandb-watch-not-logging-parameters/1197/16
        (
            predictions,
            attns,
            _,
        ) = self(
            x=x,
        )

        loss = F.cross_entropy(predictions[:, -1, :], y)
        accuracy = (torch.argmax(predictions[:, -1, :], dim=-1) == y).float().mean()
        attn_entropies = sum([-(attn * torch.log(attn + 1e-7)).sum(dim=-1).mean().item() for attn in attns]) / len(
            attns
        )
        param_norm = parameter_norm(self)

        return (
            loss,
            {
                "loss": (loss.item(), x.shape[0]),
                "accuracy": (accuracy.item(), x.shape[0]),
                "attn_entropy": (
                    attn_entropies,
                    len(attns) * x.shape[0] * (x.shape[1] - 1),
                ),
                "param_norm": (param_norm, 1),
            },
        )
