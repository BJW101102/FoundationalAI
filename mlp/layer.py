import numpy as np
from .activation import ActivationFunction
from typing import Tuple

class Layer:

    def __init__(self, fan_in: int, fan_out: int, activation_function: ActivationFunction, num: int):
        """
        Initializes a layer of neurons

        :param fan_in: number of neurons in previous (presynpatic) layer
        :param fan_out: number of neurons in this layer
        :param activation_function: instance of an ActivationFunction
        :param num: the layer's number/index
        """
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.activation_function = activation_function    
        self.num = num    
        self.activations = None
        self.Z = None
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

        z = np.dot(h, self.W) + self.b
        self.Z = z
        self.activations = self.activation_function.forward(z)
        return self.activations



    def backward(self, h: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply backpropagation to this layer and return the weight and bias gradients.

        :param h: input to this layer
        :param delta: delta term from the layer above
        :return: (weight gradients, bias gradients)
        """

        dO_dZ = self.activation_function.derivative(self.Z)
        dZ_dW = h
        dZ_DOP = self.W
        dL_dW = np.dot(dZ_dW.T, delta * dO_dZ)
        dL_db = np.sum(delta * dO_dZ, axis=0)
        self.delta = np.dot(delta * dO_dZ, dZ_DOP.T)

        return dL_dW, dL_db


    
