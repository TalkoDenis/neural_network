import numpy as np
from src.nn.loss import Loss

class MSE(Loss):
    def __init__(self):
        pass

    def forward(self, y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))

    def backward(self, y_true, y_pred):
        n = y_true.size
        return (2 / n) * (y_pred - y_true)
