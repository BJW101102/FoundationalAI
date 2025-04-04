import numpy as np
from abc import ABC, abstractmethod
class ActivationFunction(ABC):
    
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the output of the activation function, evaluated on x

        Input args may differ in the case of softmax

        :param x (np.ndarray): input
        :return: output of the activation function
        """
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the activation function, evaluated on x
        :param x (np.ndarray): input
        :return: activation function's derivative at x
        """
        pass

class Linear(ActivationFunction):
    def forward(self, x):
        return x
    
    def derivative(self, x):
        func = self.forward(x)
        return np.ones_like(func)

class Sigmoid(ActivationFunction):
    def forward(self, x):
        return 1 / (1+np.exp(-x))
    
    def derivative(self, x):
        y = self.forward(x)
        dy_dx = y * (1 - y) 
        return dy_dx

class Tanh(ActivationFunction):
    def forward(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
    def derivative(self, x):
        func = self.forward(x)
        dy_dx = 1 - np.square(func)
        return dy_dx

class Relu(ActivationFunction):
    def forward(self, x):
        return np.maximum(0, x)
    
    def derivative(self, x):
        return np.where(x > 0, 1, 0)

class Softplus(ActivationFunction):
    def forward(self, x):
        return np.log(1 + np.exp(x))
    
    def derivative(self, x):
        return 1 / (1 + np.exp(-x))

class Mish(ActivationFunction):
    def forward(self, x):
        softplus = Softplus().forward(x)  # Use the Softplus instance to compute the forward pass
        return x * np.tanh(softplus)

    def derivative(self, x):
        softplus = Softplus().forward(x)  # Use the Softplus instance to compute the forward pass
        tanh_term = np.tanh(softplus)
        sech_term = 1 / (np.cosh(softplus))**2
        return tanh_term + x * sech_term * (1 / (1 + np.exp(-x)))

class Softmax(ActivationFunction):
    def forward(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    def derivative(self, x):
        y = self.forward(x)  
        dy_dx = np.zeros_like(y)
        for i in range(y.shape[0]):  
            softmax_output = y[i]
            diag_y = np.diag(softmax_output)
            yyT = np.outer(softmax_output, softmax_output.T)
            jacobian = diag_y - yyT
            dy_dx[i] = np.diagonal(jacobian)  
        return dy_dx  


