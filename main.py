from src.nn.linear import Linear
from src.nn.activations import ReLU
from src.nn.losses import MSE
from src.nn.network import Network
from src.cli import parse_arguments
from src.data.dataset import XORData

args = parse_arguments()

dataset = XORData()
x_train, y_train = dataset.get_data()

my_network = Network([
                         Linear(input_size=2, output_size=4),
                         ReLU(),
                         Linear(input_size=4, output_size=3)
                     ],
                     loss_function=MSE())

epochs = args.epochs
learning_rate = args.learning_rate

print("Starting to learn...\n")

for epoch in range(epochs):
    for i in range(len(x_train)):
        x = x_train[i:i+1]
        y = y_train[i:i+1]
        
        my_network.train(x, y, learning_rate)
        
    if epoch % 100 == 0:
        total_error = 0
        for i in range(len(x_train)):
            x = x_train[i:i+1]
            y = y_train[i:i+1]
            total_error += my_network.loss.forward(y, my_network.predict(x))
        print(f"Epoch {epoch} | Average Error: {total_error / 4:.4f}")

print("\nTraining Complete! Let's look at the final predictions:")
for i in range(len(x_train)):
    x = x_train[i:i+1]
    prediction = my_network.predict(x)
    print(f"Input: {x[0]} | Target: {y_train[i][0]} | Network Guessed: {prediction[0][0]:.4f}")
