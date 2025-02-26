# Brandon Walton
# Dr. James Ghawaly
# CSC 4700 

from mlp.loss import *
from mlp.activation import *
import matplotlib.pyplot as plt
from loaders.mnist_load import load_mnist_data
from sklearn.model_selection import train_test_split
from mlp.multi_perceptron import MultilayerPerceptron, initialize_layers, plot_training_graph

PADDING = 30

def select_samples_per_class(y_test, num_classes):
    """Select one sample from each class (0-9)."""
    sample_indices = []
    labels = []
    
    for class_label in range(num_classes):
        class_idx = np.where(np.argmax(y_test, axis=1) == class_label)[0][0]
        sample_indices.append(class_idx)
        labels.append(class_label)
    
    return sample_indices, labels

def display_mnist_pred(x_test, y_test, network_output, num_classes):
    """Displays the mnist images with predicted and true values"""
    selected_indices, selected_labels = select_samples_per_class(y_test, num_classes)
    
    # Get predictions from network output
    predictions = np.argmax(network_output[selected_indices], axis=1)

    # Plot the selected images
    _, axes = plt.subplots(2, 5, figsize=(15, 6))  # 2 rows, 5 columns
    axes = axes.flatten()  

    for i in range(len(selected_indices)):
        sample_image = x_test[selected_indices[i]].reshape(28, 28)
        axes[i].imshow(sample_image, cmap='gray')
        axes[i].set_title(f'Pred: {predictions[i]}\nTrue: {selected_labels[i]}')
        axes[i].axis('off')  # Hide the axes for better clarity

    plt.tight_layout() 
    plt.show()

def accuracy(y_true, y_pred):
    """Calculate accuracy by comparing predicted and true labels."""
    predicted_labels = np.argmax(y_pred, axis=1)
    true_labels = np.argmax(y_true, axis=1)
    correct_predictions = np.sum(predicted_labels == true_labels)
    
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
          neurons_in_hidden_layer=64, 
          debug=True
     )
     
     epochs = 40
     save_dir = './images'
     save_name='mnist_bwalton.png'

     # Training MLP
     multi_p = MultilayerPerceptron(layers=layers)
     print(f'{'='*PADDING}TRAINING{'='*PADDING}')
     training_loss, validation_loss = multi_p.train(
          train_x=x_train, 
          train_y=y_train, 
          val_x=x_val,  
          val_y=y_val, 
          learning_rate=1E-3,
          batch_size=16,
          epochs=epochs,
          loss_func=loss_function
     )
     
     print(f'{'='*PADDING}TESTING{'='*PADDING}')
     network_output = multi_p.forward(x=x_test)
     acc = accuracy(y_test, network_output)
     print(f"Accuracy: {acc * 100:.2f}%")
     
     print(f'{'='*PADDING}PLOTTING{'='*PADDING}')
     plot_training_graph(epochs, training_loss, validation_loss, save_dir, save_name)
     
     print(f'{'='*PADDING}EXAMPLES{'='*PADDING}')
     display_mnist_pred(x_test, y_test, network_output, output_size)
          

