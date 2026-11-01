import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        x = np.asarray(x) # (N,)
        input_x = x.copy()
        for curr_layer in range(len(biases)):
            # x : (1, N)
            # W0: (N, d)
            # biases[curr_layer]: (d,)

            # x[None, :]            # (1,N)
            # weights[curr_layer]   # (N, d)


            y_hat = (weights[curr_layer].T @ input_x[:, None]) + biases[curr_layer][:,None]

            if curr_layer != len(biases)-1:
                y_hat = np.maximum(0, y_hat)

            input_x = y_hat.reshape(-1)

        return input_x
