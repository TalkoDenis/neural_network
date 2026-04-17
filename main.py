from src.nn.linear import Linear
from src.nn.activations import ReLU
from src.nn.losses import MSE
from src.nn.network import Network
from src.cli import parse_arguments
from src.data.dataset import XORData
from src.nn.trainer import Trainer

args = parse_arguments()

dataset = XORData()
x_train, y_train = dataset.get_data()

my_network = Network([
                         Linear(input_size=2, output_size=4),
                         ReLU(),
                         Linear(input_size=4, output_size=3)
                     ],
                     loss_function=MSE())


trainer = Trainer(network=my_network, learning_rate=args.learning_rate)
trainer.train(x_train, y_train, epochs=args.epochs)
