import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        eq_1 = (y_true == 1)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        ln_arr = np.zeros_like(y_pred)
        ln_arr[eq_1] = np.log(y_pred[eq_1])
        ln_arr[~eq_1] = np.log(1 - y_pred[~eq_1])
        return round((-1.0 * np.sum(ln_arr) / y_true.shape[0]), 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        eq_1 = (y_true == 1)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        loss = np.zeros_like(y_pred)
        loss[eq_1] = np.log(y_pred[eq_1])
        # loss[~eq_1] = np.log(1 - y_pred[~eq_1])
        return round(-np.sum(loss) / y_true.shape[0], 4)
