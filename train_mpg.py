# Brandon Walton
# Dr. James Ghawaly
# CSC 4700 

import numpy as np
import pandas as pd
from mlp.loss import *
from mlp.activation import *
from loaders.mpg_load import load_mpg_dataset
from mlp.multi_perceptron import MultilayerPerceptron, initialize_layers, plot_training_graph

PADDING = 30

def select_random_samples(x_test, y_test, num_samples=10):
    """Select 10 random samples from the test set."""
    random_indices = np.random.choice(len(x_test), num_samples, replace=False)
    return x_test[random_indices], y_test[random_indices]

def display_mpg_predictions(x_test, y_test, network_output):
    """Displays predicted MPG against true MPG in a table."""
    selected_samples, selected_true_labels = select_random_samples(x_test, y_test)
    
    # Get indices of the selected samples (since x_test and network_output should have matching shapes)
    selected_indices = np.array([np.where(np.all(x_test == sample, axis=1))[0][0] for sample in selected_samples])
    
    # Use selected_indices to get the corresponding predictions
    predictions = network_output[selected_indices]
    
    results_df = pd.DataFrame({
        'True MPG': selected_true_labels.flatten(),
        'Predicted MPG': predictions.flatten()
    })

    print("\nPredicted MPG vs True MPG:")
    print(results_df)


    
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
          neurons_in_hidden_layer=128, 
          debug=True
     )

     # Training MLP
     
     epochs = 100
     save_dir = './images'
     save_name='mpg_bwalton.png'
     dataset = 'MPG'


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
     network_output = multi_p.forward(x=x_test)
     testing_loss = loss_function.loss(y_true=y_test, y_pred=network_output)
     
     print(f'Testing Loss: {np.mean(testing_loss)}')
     
     print(f'{'='*PADDING}PLOTTING{'='*PADDING}')
     plot_training_graph(epochs, training_loss, validation_loss, dataset, save_dir, save_name)
     
     print(f'{"="*PADDING}PREDICTIONS{"="*PADDING}')
     display_mpg_predictions(x_test, y_test, network_output)
     
