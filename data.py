from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import openml
import pandas as pd
from openml import OpenMLDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch 
from config import SEED

task_ids = (
    40588,
    40589,
    40590,
    40591,
    40592,
    40593,
    40594,
    40595,
    40596,
    40597,
)


@dataclass
class Split:
    X: np.ndarray
    y: np.ndarray


@dataclass
class Dataset:
    name: str
    id: int
    features: pd.DataFrame
    labels: pd.DataFrame
    openml: OpenMLDataset
    encoders: dict[str, LabelEncoder]

    def split(
        self,
        splits: Iterable[float],
        seed: int | None = 1,
    ) -> tuple[Split, ...]:
        """Create splits of the data

        Parameters
        ----------
        splits : Iterable[float]
            The percentages of splits to generate

        seed : int | None = None
            The seed to use for the splits

        Returns
        -------
        tuple[Split, ...]
            The collected splits
        """
        splits = list(splits)
        assert abs(1 - sum(splits)) <= 1e-6, "Splits must sum to 1"

        sample_sizes = tuple(int(s * len(self.features)) for s in splits)

        collected_splits = []

        next_xs = self.features.to_numpy()
        next_ys = self.labels.to_numpy()

        for size in sample_sizes[:-1]:
            xs, next_xs, ys, next_ys = train_test_split(
                next_xs, next_ys, train_size=size, random_state=SEED
            )
            collected_splits.append(Split(X=xs, y=ys))
        collected_splits.append(Split(X=next_xs, y=next_ys))

        return tuple(collected_splits)

    @staticmethod
    def from_openml(id: int) -> Dataset:
        """Processes an multilabel OpenMLDataset into its features and targets

        Parameters
        ----------
        id: int
            The id of the dataset

        Returns
        -------
        Dataset
        """
        dataset = openml.datasets.get_dataset(id)
        print(dataset.name, id)
        targets = dataset.default_target_attribute.split(",")
        data, _, _, _ = dataset.get_data()

        assert isinstance(data, pd.DataFrame)

        # Process the features and turn all categorical columns into ints
        features = data.drop(columns=targets)
        encoders: dict[str, LabelEncoder] = {}

        for name, col in features.iteritems():
            if col.dtype in ["object", "category", "string"]:
                encoder = LabelEncoder()
                features[name] = encoder.fit_transform(col)
                encoders[name] = encoder

        labels = data[targets]

        # Since we assume binary multilabel data, we convert the labels
        # to all be boolean types
        labels = labels.astype(bool)

        return Dataset(
            name=dataset.name,
            id=id,
            features=features,
            labels=labels,
            openml=dataset,
            encoders=encoders,
        )


class MyDataset(Dataset):
 
  def __init__(self,split):
 
    x=np.array(split.X)
    y=np.array(split.y)
 
    self.x_train=torch.tensor(x,dtype=torch.float32)
    self.y_train=torch.tensor(y,dtype=torch.float32)
 
  def __len__(self):
    return len(self.y_train)
   
  def __getitem__(self,idx):
    return self.x_train[idx],self.y_train[idx]



if __name__ == "__main__":
    # Open the first dataset in a browser
    first = task_ids[0]
    dataset = Dataset.from_openml(first)
    dataset.openml.open_in_browser()

    train, val, test = dataset.split(splits=(0.6, 0.2, 0.2))
    print(train.X.shape, val.X.shape, test.X.shape)
