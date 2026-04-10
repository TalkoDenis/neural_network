import numpy as np
from src.nn.linear import Linear
from src.nn.activations import ReLU
from src.nn.network import Network

my_data = np.array([[0.8, -0.2]])

my_network = Network([
                         Linear(input_size=2, output_size=4),
                         ReLU(),
                         Linear(input_size=4, output_size=3)
                     ])

final_answer = my_network.predict(my_data)

print(f'Input data {my_data}')
print(f'Predict {final_answer}')
