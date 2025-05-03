import torch
from grokking.grokk_replica.grokk_model import GrokkModel


MODEL_CLASS = GrokkModel
LR_SCHEDULER_CLASS = torch.optim.lr_scheduler.SequentialLR
OPTIMIZER_CLASS = torch.optim.AdamW
