import numpy as np
from src.nn.factory import NetworkFactory
from src.data.dataset import RealData
from src.cli import parse_inference_args

args = parse_inference_args()
dataset = RealData(filepath='src/data/houses.csv', target_column='price')

my_network = NetworkFactory.create('deep')
my_network.load('models/model_brain.pkl')

new_houses = np.array([[args.size, args.bedrooms]])
new_houses_squished = my_network.predict(new_houses_squished)

prediction_squished = my_network.predict(new_houses_squished)

real_price = prediction_squished[0][0] * (dataset.y_max[0] + 1e-8)

print(f'Size {args.size}')
print(f'bedrooms {args.bedrooms}')
print(f'price {args.price}')
