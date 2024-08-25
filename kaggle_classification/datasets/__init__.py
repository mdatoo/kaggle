"""PyTorch datasets."""

from .catsdogs_dataset import CatsdogsDataset
from .classification_dataset import ClassificationDataset
from .petals_dataset import PetalsDataset

__all__ = ["CatsdogsDataset", "ClassificationDataset", "PetalsDataset"]
