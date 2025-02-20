import random
import numpy as np
from mnist_load import load_mnist_data
import matplotlib.pyplot as plt
from mlp.loss import *
from mlp.activation import *
from mlp.multi_perceptron import MultilayerPerceptron, Layer

            
def initialize_layers(input_size: int, output_size: int, input_activation: ActivationFunction, hidden_activation: ActivationFunction, output_activation: ActivationFunction, num_hidden_layers: int=1, neurons_in_hidden_layer: int=32, debug: bool=False) -> list[Layer]:
        """
        Initialize layers for the neural network based on the given configuration.
        :param layer_config: List of tuples where each tuple is (fan_in, fan_out, activation_function)
        :return: List of Layer instances
        """
        
        layers = []
        
        # Generate a random number of hidden layers
        num_hidden_layers = num_hidden_layers
        
        # The middle of the input and output size
        neurons_in_hidden_layer = neurons_in_hidden_layer
        
        # Creating Input Layer
        input_layer = Layer(input_size, neurons_in_hidden_layer, input_activation)
        layers.append(input_layer)
                
        # Creating Hidden Layers
        hidden =""
        for _ in range(num_hidden_layers):
            fan_in = neurons_in_hidden_layer
            fan_out = max(neurons_in_hidden_layer // 2, 1)
            layer = Layer(fan_in, fan_out, hidden_activation)
            hidden +=f'- Hidden: # of Neurons={fan_in}\n'
            neurons_in_hidden_layer = fan_out
            layers.append(layer)
            
        # Creating Output Layers
        output_layer = Layer(neurons_in_hidden_layer, output_size, output_activation)
        layers.append(output_layer)
        
        if debug:
            total_layers = num_hidden_layers + 2
            print(f'Creating Network with {total_layers} Layers')
            print(f'- Input: # of Neurons={input_size}\n' + hidden + f'- Output: # of Neurons={output_size}')

        return layers

# Gradient: partial derivative of a function in respect to all of its independent variables 
# Gradient Descent: optimization algorithm used to minimize some function by iteratively moving in the direction of steepest descent as defined by the negative of the gradient
# Backpropagation: a method used to calculate the gradient of the loss function with respect to the weights of the network
# Loss Function: a method of evaluating how well the algorithm models the dataset
# Activation Function: a function that determines the output of a neural network
# Epoch: one forward pass and one backward pass of all the training examples
# Batch Size: the number of training examples utilized in one iteration
# Learning Rate: a hyperparameter that controls how much we are adjusting the weights of our network with respect to the loss gradient
# Momentum: a method that helps accelerate gradients vectors in the right directions, thus leading to faster converging
# Regularization: a technique used to reduce overfitting by discouraging overly complex models in some way
# Dropout: a technique used to prevent overfitting by randomly setting some neurons to zero during forward and backward pass
# Overfitting: a model that is too complex and fits the training data too well
# Underfitting: a model that is too simple and does not fit the training data well
# Hyperparameter: a parameter whose value is set before the learning process begins

if __name__ == '__main__':
    debug = True
    (x_train_full, y_train_full), (x_test, y_test) = load_mnist_data()

    split_idx = int(0.8 * len(x_train_full))
    x_train = x_train_full[:split_idx]
    y_train = y_train_full[:split_idx]
    x_val = x_train_full[split_idx:]
    y_val = y_train_full[split_idx:]

    # Gathering Metrics
    input_size = len(x_train[0])
    output_size= len((y_train[0]))

    # Initializing Network Layers
    input_activation_function = Linear()
    hidden_activation_function = Relu()
    output_activation_function = Softmax()
    loss_function = CrossEntropy()

    layers = initialize_layers(
         input_size=input_size, 
         output_size=output_size, 
         input_activation=input_activation_function, 
         hidden_activation=hidden_activation_function,
         output_activation=output_activation_function,
         num_hidden_layers=1,
         neurons_in_hidden_layer=16, 
         debug=debug
    )

    # Training Network
    multi_p = MultilayerPerceptron(layers=layers)
    multi_p.train(
         train_x=x_train, 
         train_y=y_train, 
         val_x=x_val,  
         val_y=y_val, 
         loss_func=loss_function,
        )
    



