import numpy as np

class XORData():
    def __init__(self):
        self.x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y_train = np.array([[0], [1], [1], [0]])

    def get_data(self):
        return self.x_train, self.y_train
