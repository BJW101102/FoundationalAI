import numpy as np
from typing import Tuple
from .layer import Layer
from .loss import LossFunction

import numpy as np

def batch_generator(train_x, train_y, batch_size):
    """
    A function that returns a list of mini-batches of data.
    
    :param train_x: Features of shape (n, f) where 'n' is the number of samples and 'f' is the number of features.
    :param train_y: Labels of shape (n, q) where 'n' is the number of samples and 'q' is the number of target classes.
    :param batch_size: The size of each mini-batch.
    
    :return: A list of tuples [(batch_x, batch_y), (batch_x, batch_y), ...] for each mini-batch.
    """

    n_samples = len(train_x)  # Get the total number of samples.

    indices = np.arange(n_samples)  # Create an array of indices (0, 1, ..., n-1).
    
    
    batches = []  # Create an empty list to hold the batches.

    # Loop through the indices in steps of batch_size.
    for start_idx in range(0, n_samples, batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]  # Get indices for the current batch.
        
        # Ensure batch_indices is of correct integer type
        batch_x = train_x[batch_indices]  # Get the input data for the batch.
        batch_y = train_y[batch_indices]  # Get the target data for the batch.
        
        batches.append((batch_x, batch_y))  # Add the batch to the list of batches.

    return batches  # Return the list of all batches.


class MultilayerPerceptron:
    def __init__(self, layers: Tuple[Layer] | list[Layer]):
        """
        Create a multilayer perceptron (densely connected multilayer neural network)
        :param layers: list or Tuple of layers
        """
        self.layers = layers
        
    # Finished
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        This takes the network input and computes the network output (forward propagation)
        :param network_input: network input
        :return: network output
        """
        
        for layer in self.layers:
            x = layer.forward(h=x)

        return x

    def backward(self, loss_grad: np.ndarray, input_data: np.ndarray) -> Tuple[list, list]:

        # gradient: vector of derivatives of losses 

        """
        Applies backpropagation to compute the gradients of the weights and biases for all layers in the network
        :param loss_grad: gradient of the loss function
        :param input_data: network's input data
        :return: (List of weight gradients for all layers, List of bias gradients for all layers)
        """
        dl_dw_all = []
        dl_db_all = []


        delta = loss_grad
        for i, layer in enumerate(reversed(self.layers)):
            
            if i == 0:
                prev_activation = input_data  # The input to the network is the first layer's "h".
            else:
                prev_activation = self.layers[-(i+1)].activations # Activations from the previous layer.
            
            dL_dW, dL_db = layer.backward(h=prev_activation, delta=delta)
            

            # Partial derivate of the input with respect to the previous layer's output
            dZ_dPO = layer.W

            # delta = np.dot(layer.delta,)

            dl_dw_all.append(dL_dW)
            dl_db_all.append(dL_db)

    


        return None, None

    def train(self, train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, loss_func: LossFunction, learning_rate: float=1E-3, batch_size: int=16, epochs: int=32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train the multilayer perceptron

        :param train_x: full training set input of shape (n x d) n = number of samples, d = number of features
        :param train_y: full training set output of shape (n x q) n = number of samples, q = number of outputs per sample
        :param val_x: full validation set input
        :param val_y: full validation set output
        :param loss_func: instance of a LossFunction
        :param learning_rate: learning rate for parameter updates
        :param batch_size: size of each batch
        :param epochs: number of epochs
        :return:
        """

        training_losses = None
        validation_losses = None

        batches = batch_generator(train_x, train_y, batch_size)

        for epoch in range(epochs):
            for batch_x, batch_y in batches:
                
                network_input = np.array(batch_x) # arrays of input data
                y_true = np.array(batch_y)
                # Step 1. Forward pass
                network_output = self.forward(network_input) # arrays of output data

                # Step 2. Compute the loss
                loss = loss_func.loss(y_true=y_true, y_pred=network_output) 

                # Step 3. Compute loss gradient
                loss_grad = loss_func.derivative(y_true=y_true, y_pred=network_output)

                # Step 4. Backward pass
                


            break




        return training_losses, validation_losses