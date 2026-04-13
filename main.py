import numpy as np
from src.nn.linear import Linear
from src.nn.activations import ReLU
from src.nn.losses import MSE
from src.nn.network import Network

x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

y_train = np.array([[0], [1], [1], [0]])

my_network = Network([
                         Linear(input_size=2, output_size=4),
                         ReLU(),
                         Linear(input_size=4, output_size=3)
                     ],
                     loss_function=MSE())

epochs = 1000
learning_rate = 0.1

print("Starting to learn...\n")

for epoch in range(epochs):
    # We train the network on one example at a time
    for i in range(len(x_train)):
        # Extract one row of data and keep it as a matrix
        x = x_train[i:i+1]
        y = y_train[i:i+1]
        
        # Train the network and update the weights!
        my_network.train(x, y, learning_rate)
        
    # Every 100 epochs, let's see how much the error has gone down!
    if epoch % 100 == 0:
        total_error = 0
        for i in range(len(x_train)):
            x = x_train[i:i+1]
            y = y_train[i:i+1]
            total_error += my_network.loss.forward(y, my_network.predict(x))
        print(f"Epoch {epoch} | Average Error: {total_error / 4:.4f}")

# 4. The Final Test!
print("\nTraining Complete! Let's look at the final predictions:")
for i in range(len(x_train)):
    x = x_train[i:i+1]
    prediction = my_network.predict(x)
    print(f"Input: {x[0]} | Target: {y_train[i][0]} | Network Guessed: {prediction[0][0]:.4f}")
