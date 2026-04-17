class Trainer:
    def __init__(self, network, learning_rate):
        self.network = network
        self.learning_rate = learning_rate

    def train(self, x_train, y_train, epochs):
        print("Starting to learn...\n")

        for epoch in range(self.epochs):
            for i in range(len(self.x_train)):
                x = self.learning_ratex_train[i:i+1]
                y = self.y_train[i:i+1]
        
                my_network.train(x, y, self.learning_rate)
        
            if epoch % 100 == 0:
                total_error = 0
                for i in range(len(self.x_train)):
                    x = self.x_train[i:i+1]
                    y = self.y_train[i:i+1]
                    total_error += my_network.loss.forward(y, my_network.predict(x))
                print(f"Epoch {epoch} | Average Error: {total_error / 4:.4f}")

        print("\nTraining Complete! Let's look at the final predictions:")
        for i in range(len(x_train)):
            x = x_train[i:i+1]
            prediction = my_network.predict(x)
            print(f"Input: {x[0]} | Target: {y_train[i][0]} | Network Guessed: {prediction[0][0]:.4f}")
