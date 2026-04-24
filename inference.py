import numpy as np
import sys
from src.nn.factory import NetworkFactory
from src.data.dataset import RealData
from src.cli import parse_inference_args

args = parse_inference_args()
try:
    dataset = RealData(filepath=args.data, target_column=args.target)
except FileNotFoundError:
    print(f"Error! Could not find a dataset at {args.data}")
    sys.exit(1)

my_network = NetworkFactory.create("deep")
try:
    my_network.load("models/model_brain.pkl")
except FileNotFoundError:
    print('Error! Could not find ".plk" file')
    sys.exit(1)

new_data = np.array([args.features])

new_data_squished = new_data / (dataset.x_max + 1e-8)

prediction_squished = my_network.predict(new_data_squished)

real_prediction = prediction_squished[0][0] * (dataset.y_max[0] + 1e-8)

print(f"Inputs: {args.features}")
print(f"Target column: {args.target}")
print(f"Result {real_prediction}")
