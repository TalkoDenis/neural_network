import numpy as np
from src.nn.losses import MSE

def test_mse_loss():
    y_true = np.array([[1.0]])
    y_pred = np.array([[0.5]])

    loss_function = MSE()
    error = loss_function.forward(y_true, y_pred)
    
    assert error == 0.25
