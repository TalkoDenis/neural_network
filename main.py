from src.cli import parse_arguments
from src.data.dataset import RealData
from src.nn.trainer import Trainer
from src.nn.factory import NetworkFactory

args = parse_arguments()

dataset = RealData(filepath=args.data, target_colums=args.target)
x_train, y_train = dataset.get_data()

my_network = NetworkFactory.create(args.network)

trainer = Trainer(network=my_network, learning_rate=args.learning_rate)
trainer.train(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size)
