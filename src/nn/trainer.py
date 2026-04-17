class Trainer:
    def __init__(self, network, learning_rate):
        self.network = network
        self.learning_rate = learning_rate

    def train(self, x_train, y_train, epochs):
        print("Starting to learn...\n")

        for epoch in range(epochs):
            for i in range(len(x_train)):
                x = x_train[i:i+1]
                y = y_train[i:i+1]
        
                self.network.train(x, y, self.learning_rate)
        
            if epoch % 100 == 0:
                total_error = 0
                for i in range(len(x_train)):
                    x = x_train[i:i+1]
                    y = y_train[i:i+1]
                    total_error += self.network.loss.forward(y, self.network.predict(x))
                print(f"Epoch {epoch} | Average Error: {total_error / 4:.4f}")

        print("\nTraining Complete! Let's look at the final predictions:")
        for i in range(len(x_train)):
            x = x_train[i:i+1]
            prediction = self.network.predict(x)
            print(f"Input: {x[0]} | Target: {y_train[i][0]} | Network Guessed: {prediction[0][0]:.4f}")
