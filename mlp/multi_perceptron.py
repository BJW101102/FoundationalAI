import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
from .layer import Layer
from .loss import LossFunction

def batch_generator(train_x, train_y, batch_size):
    """
    A function that returns a list of mini-batches of data.
    
    :param train_x: Features of shape (n, f) where 'n' is the number of samples and 'f' is the number of features.
    :param train_y: Labels of shape (n, q) where 'n' is the number of samples and 'q' is the number of target classes.
    :param batch_size: The size of each mini-batch.
    
    :return: A list of tuples [(batch_x, batch_y), (batch_x, batch_y), ...] for each mini-batch.
    """

    n_samples = len(train_x) 
    indices = np.arange(n_samples)  
    batches = []  

    # Loop through the indices in steps of batch_size.
    for start_idx in range(0, n_samples, batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]  # Get indices for the current batch.
        
        # Ensure batch_indices is of correct integer type
        batch_x = train_x[batch_indices]  
        batch_y = train_y[batch_indices]  
        
        batches.append((batch_x, batch_y))  

    return batches  # Return the list of all batches.

import matplotlib.pyplot as plt
import os

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
        
        for i, layer in enumerate(self.layers):
            x = layer.forward(h=x)

        return x

    def backward(self, loss_grad: np.ndarray, input_data: np.ndarray) -> Tuple[list, list]:
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
            
            input_layer_index = len(self.layers) - 1
            current_layer_index = input_layer_index - i
            prev_layer_index = current_layer_index - 1
            
            if i == input_layer_index:
                prev_activation = input_data  # The input to the network is the first layer's "h".
            else:
                prev_activation = self.layers[prev_layer_index].activations # Activations from the previous layer.
            
            # Compute gradients for weights and biases
            dL_dW, dL_db = layer.backward(h=prev_activation, delta=delta)
            
            # Partial derivatives of the layer's activations
            dZ_dPO = layer.W
            dO_dZ = layer.activation_function.derivative(layer.activations)       
            
            # Update delta for backpropagation
            delta = np.dot(layer.delta * dO_dZ, dZ_dPO.T)

            dl_dw_all.append(dL_dW)
            dl_db_all.append(dL_db)

        dl_dw_all.reverse()
        dl_db_all.reverse()

        return dl_dw_all, dl_db_all

    def train(self, train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, loss_func: LossFunction, learning_rate: float=1E-3, batch_size: int=16, epochs: int=32, save_dir: str = "./") -> Tuple[np.ndarray, np.ndarray]:
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
        :param save_dir: directory to save the plot image (default is current directory)
        :return:
        """
        
        training_losses = []
        validation_losses = []

        batches = batch_generator(train_x, train_y, batch_size)
        total_batches = len(batches)

        for epoch in range(epochs):
            epoch_training_loss = 0
            epoch_validation_loss = 0
            for batch_num, (batch_x, batch_y) in enumerate(batches):
                
                # Step 0. Prepare data
                network_input = np.array(batch_x) # arrays of input data
                y_true = np.array(batch_y)
                
                # Step 1. Forward pass
                network_output = self.forward(network_input) # arrays of output data

                # Step 2. Compute loss gradient
                loss_grad = loss_func.derivative(y_true=y_true, y_pred=network_output)

                # Step 3. Backpropagation
                dl_dw_all, dl_db_all = self.backward(loss_grad=loss_grad, input_data=network_input)

                # Step 4. Update weights and biases (per layer)
                for i, layer in enumerate(self.layers):                       
                    layer.W -= learning_rate * dl_dw_all[i]  # Update weights
                    layer.b -= learning_rate * dl_db_all[i]  # Update biases

                # Step 5. Run Forward
                updated_network_output = self.forward(network_input)

                # Step 6. Compute loss
                loss = loss_func.loss(y_true=y_true, y_pred=updated_network_output)
                batch_loss = np.sum(loss)

                epoch_training_loss += batch_loss

            validated_network_output = self.forward(val_x)
            test_loss = loss_func.loss(val_y, validated_network_output)
            epoch_validation_loss += test_loss
            epoch_training_loss /= len(batches)

            # Collecting loss values for plotting
            training_losses.append(epoch_training_loss)
            validation_losses.append(epoch_validation_loss)

            print(f"Epoch {epoch+1}/{epochs} | Training Loss: {epoch_training_loss} | Validation Loss: {epoch_validation_loss}")

        # Plotting the loss values
        plt.plot(range(epochs), training_losses, label="Training Loss")
        plt.plot(range(epochs), validation_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss over Epochs")
        plt.legend()

        # Save the plot to a file in the specified directory
        os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
        plot_file = os.path.join(save_dir, "loss_plot.png")
        plt.savefig(plot_file)
        print(f"Plot saved to {plot_file}")

        # Show the plot (optional)
        plt.show()

        return training_losses, validation_losses
