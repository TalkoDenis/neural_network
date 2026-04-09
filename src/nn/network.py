class Network:
    def __init__(self, layer):
        self.layer = layer

    def predict(self, input_data):
        current_data = input_data
        for layer in self.layer:
            current_data = layer.forward(current_data)
        return current_data


