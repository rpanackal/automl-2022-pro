from sklearn.metrics import f1_score as sklearn_f1_score
import numpy as np

def f1_score(y_true: np.ndarray, y_pred:np.ndarray) -> float:
    return sklearn_f1_score(y_true, y_pred, average="macro", zero_division=0)
