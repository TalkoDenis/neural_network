from abc import ABC, abstractmethod

class Loss(ABC):

    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass
    
