import numpy as np
from src.nn.loss import Loss

class MSE(Loss):
    def __init__(self):
        pass

    def forward():
        return np.mean(np.power(y_true - y_pred, 2))

    def backwerd():
        n = y_true.size
        return (2 / n) * (y_pred - y_true)
