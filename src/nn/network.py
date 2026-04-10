class Network:
    def __init__(self, layers):
        self.layers = layers

    def predict(self, input_data):
        current_data = input_data
        for layer in self.layers:
            current_data = layer.forward(current_data)
        return current_data


