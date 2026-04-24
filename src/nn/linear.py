import numpy as np
from src.nn.layer import Layer


class Linear(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def forward(self, input_data):
        self.input = input_data
        self.output = input_data @ self.weights + self.bias
        return self.output

    def backward(self, output_error, learning_rate):
        input_error = output_error @ self.weights.T
        weights_error = self.input.T @ output_error
        bias_error = np.sum(output_error, axis=0, keepdims=True)

        self.weights = self.weights - (learning_rate * weights_error)

        self.bias = self.bias - (learning_rate * bias_error)
        return input_error
