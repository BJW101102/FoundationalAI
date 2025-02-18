
import numpy as np
from abc import ABC, abstractmethod

class LossFunction(ABC):
    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass


 # Finished
class SquaredError(LossFunction):
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return 0.5 * (y_true - y_pred) ** 2 
        
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return y_pred - y_true
 
    
# CALCULATE LATER!!!!
class CrossEntropy(LossFunction):
    pass

