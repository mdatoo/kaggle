from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
from albumentations import BaseCompose
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from dataset.image_dataset import ImageDataset


class ClassificationDataset(ImageDataset):
    def __init__(
        self, image_folder: str, labels: Dict[str, Any], transform: BaseCompose = None
    ) -> None:
        super().__init__(image_folder, transform)
        self.labels = labels

    def __getitem__(self, idx: int) -> Tuple[Union[np.ndarray, torch.Tensor], Any]:
        image = super().__getitem__(idx)
        label = self.labels[self.image_names[idx]]

        return image, label

    def split_train_test(self, test_size: float, random_state: int = None) -> Tuple[Subset, Subset]:
        train_indices, test_indices = train_test_split(
            range(len(self)),
            test_size=test_size,
            stratify=list(self.labels.values()),
            random_state=random_state,
        )

        train_dataset = Subset(self, train_indices)
        test_dataset = Subset(self, test_indices)

        return train_dataset, test_dataset
