import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from .layer import Layer
from .loss import LossFunction
from .activation import ActivationFunction

def plot_training_graph(epochs: int, training_loss: str, validation_loss:str, save_dir: str, save_name: str):
    plt.plot(range(epochs), training_loss, label="Training Loss")
    plt.plot(range(epochs), validation_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss over Epochs")
    plt.legend()
    os.makedirs(save_dir, exist_ok=True)  
    plot_file = os.path.join(save_dir, save_name)
    plt.savefig(plot_file)
    print(f"Plot saved to {plot_file}")
    plt.show()

def initialize_layers(input_size: int, output_size: int, input_activation: ActivationFunction, hidden_activation: ActivationFunction, output_activation: ActivationFunction, num_hidden_layers: int=1, neurons_in_hidden_layer: int=32, debug: bool=False) -> list[Layer]:
        """
        Initialize layers for the neural network based on the given configuration.
        :param layer_config: List of tuples where each tuple is (fan_in, fan_out, activation_function)
        :return: List of Layer instances
        """
        
        layers = []

        # Creating Input Layer
        input_layer = Layer(fan_in=input_size, fan_out=neurons_in_hidden_layer, activation_function=input_activation, num=0)
        layers.append(input_layer)

        # Creating Hidden Layers
        current_fan_in = neurons_in_hidden_layer 
        current_fan_out = neurons_in_hidden_layer  
        for i in range(num_hidden_layers):
            layer = Layer(current_fan_in, current_fan_out, hidden_activation, i + 1)
            layers.append(layer)
            
            # Update for the next hidden layer:
            current_fan_in = current_fan_out
            current_fan_out = current_fan_out // 2 

        # Create the output layer
        output_layer = Layer(current_fan_in, output_size, output_activation, num_hidden_layers + 1)
        layers.append(output_layer)

        if debug:
            print(f'Creating Network with {len(layers)} Layers')
            for i, layer in enumerate(layers):
                print(f'Layer {i}:')
                print(f'  - Fan-in: {layer.fan_in}')
                print(f'  - Fan-out: {layer.fan_out}')
                print(f'  - Neurons: {layer.fan_out}') 

        return layers


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
        batch_indices = indices[start_idx:start_idx + batch_size]  
        
        # Ensure batch_indices is of correct integer type
        batch_x = train_x[batch_indices]  
        batch_y = train_y[batch_indices]  
        
        batches.append((batch_x, batch_y))  

    return batches  


class MultilayerPerceptron:
    def __init__(self, layers: Tuple[Layer] | list[Layer]):
        """
        Create a multilayer perceptron (densely connected multilayer neural network)
        :param layers: list or Tuple of layers
        """
        self.layers = layers
        
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
        """
        Applies backpropagation to compute the gradients of the weights and biases for all layers in the network
        :param loss_grad: gradient of the loss function
        :param input_data: network's input data
        :return: (List of weight gradients for all layers, List of bias gradients for all layers)
        """
        dl_dw_all = []
        dl_db_all = []

        for k, layer in enumerate(reversed(self.layers)): 
            
            # Mapping indices 
            input_layer_index = len(self.layers) - 1
            output_layer_index = 0
            current_layer_index = input_layer_index - k
            prev_layer_index = current_layer_index - 1
            next_layer_index = current_layer_index + 1 if current_layer_index + 1 <= input_layer_index else -1

            # Grabbing the previous activations
            if k == input_layer_index:
                prev_activation = input_data   
            else:
                prev_layer = self.layers[prev_layer_index]
                prev_activation = prev_layer.activations

            # Grabbing the next layer's delta 
            if k == output_layer_index:
                delta = loss_grad
            else:
                delta = self.layers[next_layer_index].delta
                
            # Compute gradients for weights and biases
            dL_dW, dL_db = layer.backward(h=prev_activation, delta=delta)        

            dl_dw_all.append(dL_dW)
            dl_db_all.append(dL_db)

        dl_dw_all.reverse()
        dl_db_all.reverse()

        return dl_dw_all, dl_db_all

    def train(self, train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, loss_func: LossFunction, learning_rate: float=1E-3, batch_size: int=16, epochs: int=32, save_dir: str = "./", save_name:str='loss_plot.png') -> Tuple[np.ndarray, np.ndarray]:
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
            training_loss = 0

            for batch_num, (batch_x, batch_y) in enumerate(batches):
                network_input = np.array(batch_x)
                y_true = np.array(batch_y)

                # Step 1: Forward Pass
                network_output = self.forward(network_input)
                
                # Step 2: Compute Loss Gradient
                loss_grad = loss_func.derivative(y_true=y_true, y_pred=network_output)
                            
                # Step 3: Back Propagate
                dl_dw_all, dl_db_all = self.backward(loss_grad=loss_grad, input_data=network_input)
                
                # Step 4: Update Weights and Biases
                for i, layer in enumerate(self.layers):
                    layer.W -= learning_rate * dl_dw_all[i] 
                    layer.b -= learning_rate * dl_db_all[i] 
                
                # Step 5: Run Forward w/ Updated Weights
                updated_network_output = self.forward(network_input)

                # Step 6: Compute Loss for the batch (aggregate per batch)
                loss = loss_func.loss(y_true=y_true, y_pred=updated_network_output)
                training_loss += np.mean(loss)  # average over the batch

            # Run Validation on the entire validation set
            validated_network_output = self.forward(val_x)
            
            # Compute Validation Loss and aggregate (mean)
            test_loss = loss_func.loss(val_y, validated_network_output)
            validation_loss = np.mean(test_loss)

            training_loss /= total_batches

            # Track Loss
            training_losses.append(training_loss)
            validation_losses.append(validation_loss)

            print(f"Epoch {epoch+1}/{epochs} | Training Loss: {training_loss} | Validation Loss: {validation_loss}")

        return training_losses, validation_losses
