from .activation import ActivationFunction
import numpy as np
from typing import Tuple

class Layer:

    def __init__(self, fan_in: int, fan_out: int, activation_function: ActivationFunction):
        """
        Initializes a layer of neurons

        :param fan_in: number of neurons in previous (presynpatic) layer
        :param fan_out: number of neurons in this layer
        :param activation_function: instance of an ActivationFunction
        """
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.activation_function = activation_function        
        self.activations = None
        self.delta = None

        limit = np.sqrt(6 / (self.fan_in + self.fan_out)) # Glorot Initialization
        self.W = np.random.uniform(low=-limit, high=limit, size=(self.fan_in, self.fan_out))
        self.b = np.zeros((1,self.fan_out))

    def forward(self, h: np.ndarray) -> np.ndarray:
        """
        Computes the activations for this layer

        :param h: input to layer (activations from the previous layer)
        :return: layer activations
        """
        
        # Computing the weighted sum z = (W^(i)*X^(i)) + B
        z = np.dot(h, self.W) + self.b
        
        # Activating the neuron
        self.activations = self.activation_function.forward(x=z)

        return self.activations

    def backward(self, h: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply backpropagation to this layer and return the weight and bias gradients

        :param h: input to this layer
        :param delta: delta term from layer above
        :return: (weight gradients, bias gradients)
        """
        dL_dW = None 
        dL_db = None 
        self.delta = delta
        
        # Partial derivate of the Output Layer with respect to the current layer's input 
        # = Partial derivate of the layer's activations with respect to the current layer's input
        dO_dZ = self.activation_function.derivative(self.activations)       
        
        # Partial derivate of the Current Layer's input with respect to the current layer's weights
        # = Partial derivative of the previous layer's activations with respect to the current layer's weights 
        # = Previous layers activations
        dZ_dW = h

        # print('Delta Shape:', delta.shape)
        # print('dO_dZ Shape:', dO_dZ.shape)  
        # print('dZ_dW Shape:', dZ_dW.shape)
        # print('h Shape:', h.shape)  



        # Partial derivate of the Loss with respect to the current layer's weights
        dL_dW = np.dot(dZ_dW.T, delta * dO_dZ)

        # Partial derivate of the Loss with respect to the current layer's biases       
        dL_db = np.sum(delta * dO_dZ, axis=0)
                
        return dL_dW, dL_db
    
