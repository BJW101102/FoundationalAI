import numpy as np
from abc import ABC, abstractmethod

class LossFunction(ABC):
    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass

class SquaredError(LossFunction):
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return 0.5 * (y_true - y_pred) ** 2 
        
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return y_pred - y_true
     
class CrossEntropy(LossFunction):

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        epsilon = 1e-9  
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

    # Source: https://shivammehta25.github.io/posts/deriving-categorical-cross-entropy-and-softmax/
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return y_pred - y_true

