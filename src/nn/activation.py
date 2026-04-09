from src.nn.layer import Layer

class Activation(Layer):
    def __init__(self, activation_function, activation_derivative):
        super().__init__()
        self.activation = activation_function
        self.derivative = activation_derivative

    def forward(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backwards(self, output_error, learning_rate):
        pass
