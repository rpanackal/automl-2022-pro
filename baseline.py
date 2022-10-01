
from __future__ import annotations

import numpy as np
import pandas as pd
from data import Dataset, task_ids
from metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from config import SEED



class RandomForestBaseline:
    def __init__(self, seed: int | None = 1):
        self.seed = SEED
        self.estimator = RandomForestClassifier(random_state=seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.estimator.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict(X)


if __name__ == "__main__":
    seed = SEED
    scores: dict[str, float] = {}

    for id in task_ids:
        dataset = Dataset.from_openml(id)
        print(f"Training on {dataset.name}")

        train, test = dataset.split(splits=(0.80, 0.20), seed=seed)
        rf = RandomForestBaseline(seed=seed)

        rf.fit(train.X, train.y)

        predictions = rf.predict(test.X)
        score = f1_score(test.y, predictions)
        print(f"Baseline score = {score}")

        scores[dataset.name] = score

    results = pd.Series(scores)
    print(results)
