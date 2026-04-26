import numpy as np
from src.nn.linear import Linear 

def test_linear_forward_math():
    layer = Linear(input_size=2, output_size=1)
    
    layer.weights = np.array([[2.0], [3.0]])
    layer.bias = np.array([[1.0]])
    
    x_input = np.array([[4.0, 5.0]])
    
    y_output = layer.forward(x_input)
    
    assert y_output[0][0] == 24.0
