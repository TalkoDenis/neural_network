import numpy as np
import pandas as pd


class RealData:
    def __init__(self, filepath, target_column, test_size=0.2):
        if filepath.endswith(".csv"):
            df = pd.read_csv(filepath)
        elif filepath.endswith(".xlsx"):
            df = pd.read_excel(filepath)
        else:
            raise ValueError("Unsupported format!")

        x_all = df.drop(columns=[target_column]).values
        y_all = df[[target_column]].values

        split_index = int(len(x_all) * (1 - test_size))

        self.x_train = x_all[:split_index]
        self.y_train = y_all[:split_index]
        self.x_test = x_all[split_index:]
        self.y_test = y_all[split_index:]

        self.x_max = np.max(self.x_train, axis=0)
        self.y_max = np.max(self.y_train, axis=0)

        self.x_train = self.x_train / (self.x_max + 1e-8)
        self.y_train = self.y_train / (self.y_max + 1e-8)

        self.x_test = self.x_test / (self.x_max + 1e-8)
        self.y_test = self.y_test / (self.y_max + 1e-8)

    def get_data(self):
        return self.x_train, self.y_train, self.x_test, self.y_test
