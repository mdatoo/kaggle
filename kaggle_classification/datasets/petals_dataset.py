"""Petals PyTorch dataset."""

from typing import Optional

import pandas as pd
from albumentations import BaseCompose

from .classification_dataset import ClassificationDataset


class PetalsDataset(ClassificationDataset[str]):
    """Petals PyTorch dataset.

    PyTorch dataset reading from a folder of images and a csv labels file.
    """

    def __init__(self, image_folder: str, labels_file: str, transform: Optional[BaseCompose] = None) -> None:
        """Initialises object.

        Args:
            image_folder: Path to folder of images
            labels_file: Path to labels file
            transform: Image transformations to apply
        """
        labels_df = pd.read_csv(labels_file).set_index("id")

        super().__init__(image_folder, labels_df.to_dict()["label"], transform)
