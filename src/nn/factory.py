from src.nn.activations import ReLU
from src.nn.linear import Linear
from src.nn.losses import MSE
from src.nn.network import Network


class NetworkFactory:
    @staticmethod
    def create(network_type):
        if network_type == "simple":
            my_network = Network(
                [
                    Linear(input_size=2, output_size=4),
                    ReLU(),
                    Linear(input_size=4, output_size=1),
                ],
                loss_function=MSE(),
            )
        elif network_type == "deep":
            my_network = Network(
                [
                    Linear(input_size=10, output_size=16),
                    ReLU(),
                    Linear(input_size=16, output_size=8),
                    ReLU(),
                    Linear(input_size=8, output_size=1),
                ],
                loss_function=MSE(),
            )
        else:
            raise ValueError(f"Unnown network type: {network_type}")
        return my_network
