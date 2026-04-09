import numpy as np
from src.nn.activation import Activation

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(int)

class ReLU(Activation):
    def __init__(self):
        super().__init__(relu, relu_derivative)
