from mlp.loss import *
from mlp.activation import *
from mnist_load import load_mnist_data
from sklearn.model_selection import train_test_split
from mlp.multi_perceptron import MultilayerPerceptron, initialize_layers

def accuracy(y_true, y_pred):
    """Calculate accuracy by comparing predicted and true labels."""
    # Convert probabilities to predicted class labels
    predicted_labels = np.argmax(y_pred, axis=1)
    true_labels = np.argmax(y_true, axis=1)
    
    # Calculate the number of correct predictions
    correct_predictions = np.sum(predicted_labels == true_labels)
    
    # Calculate accuracy as the percentage of correct predictions
    return correct_predictions / len(y_true)

if __name__ == '__main__':
     print('loading data...')
     (x_train, y_train), (x_test, y_test) = load_mnist_data()
    
     x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

     # Gathering Metrics
     input_size = len(x_train[0])
     output_size= len((y_train[0]))

     # Initializing activation and loss functions
     input_activation_function = Linear()
     hidden_activation_function = Relu()
     output_activation_function = Softmax()
     loss_function = CrossEntropy()

     # Initializing Network Layers
     layers = initialize_layers(
          input_size=input_size, 
          output_size=output_size, 
          input_activation=input_activation_function, 
          hidden_activation=hidden_activation_function,
          output_activation=output_activation_function,
          num_hidden_layers=2,
          neurons_in_hidden_layer=32, 
          debug=True
     )

     # Training MLP
     multi_p = MultilayerPerceptron(layers=layers)
     multi_p.train(
          train_x=x_train, 
          train_y=y_train, 
          val_x=x_val,  
          val_y=y_val, 
          learning_rate=1E-3,
          batch_size=16,
          loss_func=loss_function,
          save_dir='./images',
          save_name='mnist_loss.png'
     )

     network_output = multi_p.forward(x=x_test)

     acc = accuracy(y_test, network_output)
     print(f"Accuracy: {acc * 100:.2f}%")

