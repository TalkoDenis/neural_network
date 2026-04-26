from src.nn.network import Network
from src.nn.losses import MSE

def test_netwprk_save_load(tmp_path):
    fake_layers = ['fake_layer_1', 'fake_layer_2']
    original_network = Network(layers=fake_layers, loss_function=MSE())
    test_filepath = tmp_path / 'test_brain.pki'
    original_network.save(test_filepath)
    assert test_filepath.exists() == True

    empty_network = Network(layers=[], loss_function=MSE())
    empty_network.load(test_filepath)
    assert empty_network.layers == ['fake_layer_1', 'fake_layer_2']
