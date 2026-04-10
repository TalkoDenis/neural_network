from abc import ABC, abstractmethod

class Loss(ABC):

    @abstractmethod
    def forward(self, y_true, y_pred):
        pass

    @abstractmethod
    def backward(seld, y_true, y_pred):
        pass
