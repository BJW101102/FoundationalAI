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



        # Creating Input Layer
        input_layer = Layer(fan_in=input_size, fan_out=neurons_in_hidden_layer, activation_function=input_activation, num=0)
        layers.append(input_layer)

        # Creating Hidden Layers
        current_fan_in = neurons_in_hidden_layer  # 784
        current_fan_out = neurons_in_hidden_layer  # e.g., 64

        # Create hidden layers
        for i in range(num_hidden_layers):
            # Create a hidden layer with current fan-in and fan-out
            layer = Layer(current_fan_in, current_fan_out, hidden_activation, i + 1)
            layers.append(layer)
            
            # Update for the next hidden layer:
            current_fan_in = current_fan_out
            current_fan_out = current_fan_out // 2  # or some other rule

        # Create the output layer
        output_layer = Layer(current_fan_in, output_size, output_activation, num_hidden_layers + 1)
        layers.append(output_layer)



        # Debugging Output
        if debug:
            print(f'Creating Network with {len(layers)} Layers')
            
            for i, layer in enumerate(layers):
                print(f'Layer {i+1}:')
                print(f'  - Fan-in: {layer.fan_in}')
                print(f'  - Fan-out: {layer.fan_out}')
                print(f'  - Neurons: {layer.fan_out}')  # Neurons in the layer are equal to fan-out



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
# Standardize: transforming data to have mean=0 and standard deviation=1 to ensure all features contribute equally

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

    # layers = initialize_layers(
    #      input_size=input_size, 
    #      output_size=output_size, 
    #      input_activation=input_activation_function, 
    #      hidden_activation=hidden_activation_function,
    #      output_activation=output_activation_function,
    #      num_hidden_layers=1,
    #      neurons_in_hidden_layer=128, 
    #      debug=debug
    # )



    # Training Network

    layers = [
    Layer(fan_in=input_size, fan_out=128, activation_function=Linear(), num=0),  # Input layer → Hidden Layer 1
    Layer(fan_in=128, fan_out=64, activation_function=Relu(), num=1),  # Hidden Layer 1 → Hidden Layer 2
    Layer(fan_in=64, fan_out=output_size, activation_function=Softmax(), num=2),  # Hidden Layer 1 → Hidden Layer 2

    ]


    print(type(layers))
    multi_p = MultilayerPerceptron(layers=layers)
    multi_p.train(
         train_x=x_train, 
         train_y=y_train, 
         val_x=x_val,  
         val_y=y_val, 
         learning_rate=0.0001,
         batch_size=16,
         loss_func=loss_function
    )
    



