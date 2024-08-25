"""Petals PyTorch dataset."""

from os import path
from typing import Optional

import pandas as pd
from albumentations import BaseCompose

from .classification_dataset import ClassificationDataset


class PetalsDataset(ClassificationDataset[int]):
    """Petals PyTorch dataset.

    PyTorch dataset reading from a folder of images and a csv labels file.
    """

    def __init__(self, image_folder: str, labels_file: str, transform: Optional[BaseCompose] = None) -> None:
        """Initialise object.

        Args:
            image_folder: Path to folder of images
            labels_file: Path to labels file
            transform: Image transformations to apply
        """
        image_name_to_label = pd.read_csv(labels_file).set_index("id").to_dict()["label"]
        image_path_to_label = {
            path.join(image_folder, f"{filename}.png"): label for filename, label in image_name_to_label.items()
        }

        super().__init__(image_folder, image_path_to_label, transform)
