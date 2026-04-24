import numpy as np
from src.nn.dataloader import DataLoader



def test_dataloader():
    x_dummy = np.zeros((105, 5))
    y_dummy = np.zeros((105, 1))

    dataloader = DataLoader(x_dummy, y_dummy, batch_size=10)

    batches = dataloader.get_batches()

    assert len(batches) == 11
    assert len(batches[-1][0]) == 5
