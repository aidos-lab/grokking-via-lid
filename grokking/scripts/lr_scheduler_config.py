from dataclasses import dataclass

import torch
from torch.optim.lr_scheduler import SequentialLR

from grokking.typing.enums import LRSchedulerType


@dataclass(slots=True)
class LRSchedulerConfig:
    """Configuration for the learning rate schedule."""

    lr_scheduler_type: LRSchedulerType
    warmup_steps: int
    total_steps: int

    def build(
        self,
        optimizer: torch.optim.Optimizer,
        last_step: int = -1,
    ) -> SequentialLR:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer=optimizer,
            start_factor=1.0 / 3.0,  # Note: This cannot be 0.0; we take the default value from torch.
            end_factor=1.0,
            total_iters=self.warmup_steps,
            last_epoch=last_step,
        )

        match self.lr_scheduler_type:
            case LRSchedulerType.CONSTANT:
                post_warmup_scheduler = torch.optim.lr_scheduler.ConstantLR(
                    optimizer=optimizer,
                    factor=1.0,
                    total_iters=self.total_steps - self.warmup_steps,
                    last_epoch=last_step,
                )
            case LRSchedulerType.LINEAR:
                post_warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer=optimizer,
                    start_factor=1.0,
                    end_factor=0.0,
                    total_iters=self.total_steps - self.warmup_steps,
                    last_epoch=last_step,
                )
            case _:
                msg = f"Unknown {self.lr_scheduler_type = }"
                raise ValueError(
                    msg,
                )

        result = torch.optim.lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=[
                warmup,
                post_warmup_scheduler,
            ],
            milestones=[
                self.warmup_steps,
            ],
            last_epoch=last_step,
        )

        return result
