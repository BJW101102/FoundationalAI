import random
import numpy as np
from mnist_load import load_mnist_data
import matplotlib.pyplot as plt
from mlp.loss import *
from mlp.activation import *
from mlp.multi_perceptron import MultilayerPerceptron, Layer

            
def initialize_layers(input_size: int, output_size: int, activation_function: ActivationFunction,  min_layers:int=1, max_layers:int=1, debug:bool=False) -> list[Layer]:
        """
        Initialize layers for the neural network based on the given configuration.
        :param layer_config: List of tuples where each tuple is (fan_in, fan_out, activation_function)
        :return: List of Layer instances
        """
        
        layers = []
        
        # Generate a random number of hidden layers
        num_hidden_layers = random.randint(min_layers, max_layers)
        
        # The middle of the input and output size
        neurons_in_hidden_layer = 128
        
        # Creating Input Layer
        input_layer = Layer(input_size, neurons_in_hidden_layer, activation_function)
        layers.append(input_layer)
                
        # Creating Hidden Layers
        for _ in range(num_hidden_layers):
            fan_in = neurons_in_hidden_layer
            neurons_in_hidden_layer = neurons_in_hidden_layer // 2
            fan_out = neurons_in_hidden_layer 
            layer = Layer(fan_in, fan_out,activation_function)
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
    (x_train, y_train), (x_test, y_test) = load_mnist_data()

    # Gathering Metrics
    input_size = x_train[0].reshape(-1).shape[0] 
    output_size = num_classes = len(np.unique(y_train))

    # Initializing Network Layers
    activation_function = Relu()
    loss_function = SquaredError()
    layers = initialize_layers(input_size=input_size, output_size=output_size, activation_function=activation_function, debug=debug)

    # Training Network
    multi_p = MultilayerPerceptron(layers=layers)
    multi_p.train(x_train, y_train, x_train,  y_train, loss_function)
    



