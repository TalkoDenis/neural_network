import numpy as np
import pandas as pd
from src.data.dataset import RealData

def test_dataset_normalization(tmp_path):
    fake_data = {
        'size': [100.0, 1000.0],
        'bedrooms': [1.0, 10.0],
        'price': [50000.0, 500000.0]
    }
    df = pd.DataFrame(fake_data)
    
    test_file = tmp_path / "fake_houses.csv"
    df.to_csv(test_file, index=False)
    
    dataset = RealData(filepath=str(test_file), target_column='price', test_size=0.0)
    
    x_squished = dataset.x_train[1]
    y_squished = dataset.y_train[1]
    
    assert np.isclose(x_squished[0], 1.0)
    assert np.isclose(x_squished[1], 1.0)
    assert np.isclose(y_squished[0], 1.0)
