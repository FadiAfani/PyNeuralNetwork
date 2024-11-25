from layer import Layer
import numpy as np

class Base(Layer):

    def __init__(self, input_size, output_size):
        self.bias = np.random.rand(1, output_size) - 0.5
        self.weights = np.random.rand(input_size, output_size) - 0.5
    
    def forward_propagation(self, X):
        return super().forward_propagation(X)

    def backaward_propagation(self, out_gradient, alpha):
        return out_gradient