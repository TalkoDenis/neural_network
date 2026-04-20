import numpy as np
import pandas as pd

class RealData:
    def  __init__(self, filepath, target_column):
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.xlsx'):
            df = pd.read_excel(filepath)
        else:
            raise ValueError('Unsupported format!')

        self.x_train = df.drop(columns=[target_column]).values
        self.y_train = df[[target_column]].values

        self.x_max = np.max(self.x_train, axis=0)
        self.y_max = np.max(self.y_train, axis=0)

        self.x_train = self.x_train / (self.x_max + 1e-8)
        self.y_train = self.y_train / (self.y_max + 1e-8)

    def get_data(self):
        return self.x_train, self.y_train
    
