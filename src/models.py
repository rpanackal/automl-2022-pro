from __future__ import annotations

import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from numpy import mean, std
from sklearn.datasets import make_multilabel_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score as sklearn_f1_score
from sklearn.model_selection import RepeatedKFold

import constants
from data_handling import Dataset


class RandomForestBaseline:
    def __init__(self, seed: int | None = 1):
        self.seed = seed
        self.estimator = RandomForestClassifier(random_state=seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.estimator.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict(X)


class NeuralNet():

    def __init__(self, n_inputs, n_outputs):
        """ Get the model """

        model = Sequential()
        model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
        model.add(Dense(20, kernel_initializer='he_uniform', activation='relu'))
        model.add(Dense(n_outputs, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam')
        self.model = model

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X) 


def f1_score(y_true: np.ndarray, y_pred:np.ndarray) -> float:
    return sklearn_f1_score(y_true, y_pred, average="macro", zero_division=0)


def evaluate_model(X, y):
    '''
    evaluate a model using repeated k-fold cross-validation
    '''
    results = list()
    n_inputs, n_outputs = X.shape[1], y.shape[1]
    # define evaluation procedure
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # enumerate folds
    for train_ix, test_ix in cv.split(X):
        # prepare data
        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        # define model
        model = NeuralNet(n_inputs, n_outputs)
        # fit model
        model.fit(X_train, y_train, verbose=0, epochs=100)
        # make a prediction on the test set
        yhat = model.predict(X_test)
        # round probabilities to class labels
        yhat = yhat.round()
        # calculate accuracy
        acc = f1_score(y_test, yhat)
        # store result
        print('>%.3f' % acc)
        results.append(acc)
    return results


if __name__ == "__main__":
    seed = constants.SEED
    scores: dict[str, float] = {}

    for id in constants.TASK_IDS:
        dataset = Dataset.from_openml(id)
        print(f"Training on {dataset.name}")

        train, test = dataset.split(splits=(0.75, 0.25), seed=seed)
        model = "baseline"

        if model == "baseline":
            rf = RandomForestBaseline(seed=seed)

            rf.fit(train.X, train.y)

            predictions = rf.predict(test.X)
            score = f1_score(test.y, predictions)
            print(f"Baseline score = {score}")

            scores[dataset.name] = score

        else:
            n_inputs, n_outputs = len(dataset.features.columns), len(dataset.labels.columns)

            model = NeuralNet(n_inputs, n_outputs)
            # fit model
            model.fit(train.X, train.y, verbose=0, epochs=1000)
            # make a prediction on the test set
            yhat = model.predict(test.X)
            # round probabilities to class labels
            yhat = yhat.round()
            # calculate accuracy
            score = f1_score(test.y, yhat)
            print(f"Neural network model score = {score}")

            scores[dataset.name] = score

    results = pd.Series(scores)
    print(results)
