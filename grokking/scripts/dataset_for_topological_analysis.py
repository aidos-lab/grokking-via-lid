from dataclasses import dataclass

from torch.utils.data import DataLoader

from grokking.scripts.group_dataset import GroupDataset


@dataclass
class DatasetForTopologicalAnalysis:
    """Dataset wrapper for topological analysis."""

    dataset: GroupDataset
    split: str
    dataloader: DataLoader

    def __init__(
        self,
        group_dataset: GroupDataset,
        split: str,
        train_cfg: dict,
    ) -> None:
        """Initialize DatasetForTopologicalAnalysis with dataset, split, and dataloader."""
        self.dataset = group_dataset
        self.split = split

        self.dataloader = DataLoader(
            dataset=group_dataset,
            batch_size=train_cfg["bsize"],
            shuffle=False,
            num_workers=train_cfg["num_workers"],
        )
