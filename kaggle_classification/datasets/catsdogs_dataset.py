"""Petals PyTorch dataset."""

from glob import glob
from os import path
from typing import Optional

from albumentations import BaseCompose

from .classification_dataset import ClassificationDataset


class CatsdogsDataset(ClassificationDataset[str]):
    """Petals PyTorch dataset.

    PyTorch dataset reading from a folder of images.
    """

    def __init__(self, image_folder: str, transform: Optional[BaseCompose] = None) -> None:
        """Initialise object.

        Args:
            image_folder: Path to folder of images
            transform: Image transformations to apply
        """
        image_path_to_label = {
            image_path: image_path.rsplit("/", 2)[1] for image_path in glob(path.join(image_folder, "**", "*.jpg"))
        }

        super().__init__(image_folder, image_path_to_label, transform)
