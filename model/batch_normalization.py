import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        x = np.array(x)
        # print("x.shape:", x.shape)

        if training:
            curr_mean = np.mean(x, axis=0)
            # print("curr_mean.shape: ", curr_mean.shape)

            curr_var = np.mean((x-curr_mean)**2, axis=0)
            # print("curr_var.shape: ", curr_var.shape)

            x_hat = (x - curr_mean)/np.sqrt(curr_var + eps)
            # print("x_hat.shape: ", x_hat.shape)
            # print("gamma.shape: ", len(gamma))

            y = x_hat*gamma + beta
            # print("y.shape: ", y.shape)

            running_mean = (1-momentum)*np.array(running_mean) + momentum*curr_mean
            running_var = (1-momentum)*np.array(running_var) + momentum*curr_var

            return (np.round(y, 4), np.round(running_mean, 4), np.round(running_var, 4))

        else:
            x_hat = (x - np.array(running_mean))/np.sqrt(np.array(running_var) + eps)
            y = gamma*x_hat + beta

            return (np.round(y, 4), np.round(running_mean, 4), np.round(running_var, 4))
