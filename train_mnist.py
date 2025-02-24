from mlp.loss import *
from mlp.activation import *
from mnist_load import load_mnist_data
from sklearn.model_selection import train_test_split
from mlp.multi_perceptron import MultilayerPerceptron, initialize_layers

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
          neurons_in_hidden_layer=128, 
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
          save_dir='./',
          save_name='mnist_loss.png'
     )




