import pickle


class Network:
    def __init__(self, layers, loss_function):
        self.layers = layers
        self.loss = loss_function

    def predict(self, input_data):
        current_data = input_data
        for layer in self.layers:
            current_data = layer.forward(current_data)
        return current_data

    def train(self, x_train, y_train, learning_rate):
        prediction = self.predict(x_train)
        error = self.loss.backward(y_train, prediction)
        for layer in reversed(self.layers):
            error = layer.backward(error, learning_rate)

    def save(self, filepath):
        with open(filepath, "wb") as file:
            pickle.dump(self.layers, file)

    def load(self, filepath):
        with open(filepath, "rb") as file:
            self.layers = pickle.load(file)
