
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

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        eps = 1e-10  # Small constant to prevent log(0)
        y_pred = np.clip(y_pred, eps, 1 - eps)  # Clip values to stay within [eps, 1-eps]
        return -np.sum(y_true * np.log(y_pred))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        eps = 1e-10
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -y_true / y_pred

