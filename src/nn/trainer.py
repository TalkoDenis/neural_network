from src.nn.dataloader import DataLoader


class Trainer:
    def __init__(self, network, learning_rate):
        self.network = network
        self.learning_rate = learning_rate

    def train(self, x_train, y_train, x_test, y_test, epochs, batch_size):
        print("Starting to learn...\n")

        data_loader = DataLoader(x_train, y_train, batch_size)

        for epoch in range(epochs):
            batches = data_loader.get_batches()

            for x_batch, y_batch in batches:
                self.network.train(x_batch, y_batch, self.learning_rate)

            if epoch % 100 == 0:
                prediction = self.network.predict(x_train)
                error = self.network.loss.forward(y_train, prediction)
                print(f"Epoch {epoch}, average error {error:.4f}")

        test_prediction = self.network.predict(x_test)
        test_error = self.network.loss.forward(y_test, test_prediction)

        print(f"\nTraining Complete!, Test Error is {test_error:.4f}")
        for i in range(len(x_train)):
            x = x_train[i : i + 1]
            prediction = self.network.predict(x)
            print(
                f"Input: {x[0]} | Target: {y_train[i][0]} | Network Guessed: {prediction[0][0]:.4f}"
            )
