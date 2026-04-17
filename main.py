from src.cli import parse_arguments
from src.data.dataset import XORData
from src.nn.trainer import Trainer
from src.nn.factory import NetworkFactory

args = parse_arguments()

dataset = XORData()
x_train, y_train = dataset.get_data()

my_network = NetworkFactory.create(args.network)

trainer = Trainer(network=my_network, learning_rate=args.learning_rate)
trainer.train(x_train, y_train, epochs=args.epochs)
