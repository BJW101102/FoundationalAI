
import struct
import numpy as np
import matplotlib.pyplot as plt
from array import array
from os.path import join

class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        
        images = []
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)  # Reshape image to 28x28
            images.append(img)  # Append numpy array instead of list

        return images, labels
             
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)   
    

def convert_data(x, y) -> tuple[np.ndarray, np.ndarray]:
    
    x = np.array(x)
    y = np.array(y)

    n_samples = len(x)  # Get the total number of samples.

    x = np.array([np.array(e).flatten() for e in x]) # Array of flatten arrays


    x = x / 255.0    

    print(f"min x: {np.min(x)}")
    print(f"max x: {np.max(x)}")

    # Create a matrix of zeros with shape (n_samples, num_classes)
    y_one_hot = np.zeros((n_samples, np.max(y)+1))

    y_one_hot[np.arange(n_samples), y] = 1

    return x, y_one_hot

def load_mnist_data():
    input_path = './datasets'
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    x_train, y_train = convert_data(x_train, y_train)
    x_test, y_test = convert_data(x_test, y_test)
    
    return (x_train, y_train), (x_test, y_test)