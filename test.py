import random
import numpy as np
from mnist_load import load_mnist_data
import matplotlib.pyplot as plt
from mlp.loss import *
from mlp.activation import *
from mlp.multi_perceptron import MultilayerPerceptron, Layer

def batch_generator(train_x, train_y, batch_size):
    """
    Generator that yields batches of train_x and train_y.

    :param train_x (np.ndarray): Input features of shape (n, f).
    :param train_y (np.ndarray): Target values of shape (n, q).
    :param batch_size (int): The size of each batch.

    :return tuple: (batch_x, batch_y) where batch_x has shape (B, f) and batch_y has shape (B, q). The last batch may be smaller.
    """
    pass
    
            
def initialize_layers(input_size: int, output_size: int, activation_function: ActivationFunction,  min_layers:int=1, max_layers:int=3, debug:bool=False) -> list[Layer]:
        """
        Initialize layers for the neural network based on the given configuration.
        :param layer_config: List of tuples where each tuple is (fan_in, fan_out, activation_function)
        :return: List of Layer instances
        """
        
        layers = []
        
        # Generate a random number of hidden layers
        num_hidden_layers = random.randint(min_layers, max_layers)
        
        # The middle of the input and output size
        neurons_in_hidden_layer = (input_size + output_size) // 2
        
        # Creating Input Layer
        input_layer = Layer(input_size, neurons_in_hidden_layer, activation_function)
        layers.append(input_layer)
                
        # Creating Hidden Layers
        for _ in range(num_hidden_layers):
            layer = Layer(neurons_in_hidden_layer, neurons_in_hidden_layer,activation_function)
            layers.append(layer)
            
        # Creating Output Layers
        output_layer = Layer(neurons_in_hidden_layer, output_size, activation_function)
        layers.append(output_layer)
        
        if debug:
            total_layers = num_hidden_layers + 2
            print(f'Creating Network with {total_layers} Layers')
            print(f'- Input: # of Neurons={input_size}')
            print(f'- Output: # of Neurons={output_size}')
            print(f'- Hidden: # of Neurons={neurons_in_hidden_layer}')

        return layers



if __name__ == '__main__':
    debug = True
    (x_train, y_train), (x_test, y_test) = load_mnist_data()

    # Gathering Metrics
    true_output = y_train[0]
    network_input = x_train[0].reshape(-1)  
    input_size = network_input.shape[0] 
    output_size = num_classes = len(np.unique(y_train))

    # Initializing Network Layers
    activation_function = Relu()
    layers = initialize_layers(input_size=input_size, output_size=output_size, activation_function=activation_function, debug=debug)
    
    multi_p = MultilayerPerceptron(layers=layers)

    # Perform forward pass
    network_output = multi_p.forward(x=network_input)
    
    if debug:
        y_true = np.zeros(output_size)
        y_true[true_output] = 1 
        predicted_output = np.argmax(network_output)
        sqr_err = SquaredError()
        loss = sqr_err.loss(y_true=y_true, y_pred=predicted_output)
        print(f"True Output: {true_output}")
        print(f"Predicted Output: {predicted_output}")
        print(f"Loss: {loss}")