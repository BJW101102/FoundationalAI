import os
import matplotlib.pyplot as plt
from mlp.loss import *
from mlp.activation import *
from mpg_load import load_mpg_dataset
from mlp.multi_perceptron import MultilayerPerceptron, initialize_layers, plot_training_graph

PADDING = 30

if __name__ == '__main__':
     print('loading data...')
     (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mpg_dataset()

     # Gathering Metrics
     input_size = len(x_train[0])
     output_size= len(y_train[0])

     # Initializing activation and loss functions
     input_activation_function = Linear()
     hidden_activation_function = Relu()
     output_activation_function = Linear()
     loss_function = SquaredError()

     # Initializing Network Layers
     layers = initialize_layers(
          input_size=input_size, 
          output_size=output_size, 
          input_activation=input_activation_function, 
          hidden_activation=hidden_activation_function,
          output_activation=output_activation_function,
          num_hidden_layers=2,
          neurons_in_hidden_layer=16, 
          debug=True
     )

     # Training MLP
     
     epochs = 200
     save_dir = './images'
     save_name='mpg_loss.png'

     
     multi_p = MultilayerPerceptron(layers=layers)
     print(f'{'='*PADDING}TRAINING{'='*PADDING}')
     training_loss, validation_loss = multi_p.train(
          train_x=x_train, 
          train_y=y_train, 
          val_x=x_val,  
          val_y=y_val, 
          learning_rate=1E-3,
          epochs=epochs,
          loss_func=loss_function,
          save_dir=save_dir,
          save_name=save_name
     )
     
     print(f'{'='*PADDING}TESTING{'='*PADDING}')
     network = multi_p.forward(x=x_test)
     testing_loss = loss_function.loss(y_true=y_test, y_pred=network)
     
     print(f'Testing Loss: {np.mean(testing_loss)}')
     
     print(f'{'='*PADDING}PLOTTING{'='*PADDING}')
     plot_training_graph(epochs, training_loss, validation_loss, save_dir, save_name)
     
