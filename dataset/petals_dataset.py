import pandas as pd
from albumentations import BaseCompose

from dataset.classification_dataset import ClassificationDataset


class PetalsDataset(ClassificationDataset):
    def __init__(self, image_folder: str, labels_file: str, transform: BaseCompose = None) -> None:
        labels_df = pd.read_csv(labels_file).set_index("id")

        super().__init__(image_folder, labels_df.to_dict()["label"], transform)
