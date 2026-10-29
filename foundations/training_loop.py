import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        n, d = X.shape
        w = np.zeros(d) # (d,) # (1,d)
        b = 0

        loss_history = []

        for epoch in range(epochs):
            print("w.shape: ", w.shape)
            print("X.shape: ", X.shape)
            y_hat = w @ X.T + b # (n,)
            print("y_hat.shape: ", y_hat.shape)

            mse = np.mean((y_hat - y)**2)
            loss_history.append(mse)

            error = (y_hat - y) # (n,)
            print("error.shape: ", error.shape)

            dl_dw = (2.0/n)*(error@X) # (d,)
            print("dl_dw.shape: ", dl_dw.shape)
            dl_db = (2.0/n)*(error.sum()) # ()

            w = w - lr*dl_dw
            b = b - lr*dl_db

        return (np.round(w, 5), round(b, 5))
            
