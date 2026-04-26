from src.cli import parse_arguments
from src.data.dataset import RealData
from src.nn.factory import NetworkFactory
from src.nn.trainer import Trainer

args = parse_arguments()

dataset = RealData(
    filepath=args.data, target_column=args.target, test_size=args.test_size
)
x_train, y_train, x_test, y_test = dataset.get_data()

my_network = NetworkFactory.create(args.network)

trainer = Trainer(network=my_network, learning_rate=args.learning_rate)
trainer.train(
    x_train,
    y_train,
    x_test,
    y_test,
    epochs=args.epochs,
    batch_size=args.batch_size,
)

if args.save:
    my_network.save("models/model_brain.pkl")
