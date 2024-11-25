from layer import Layer
import numpy as np

class Hidden(Layer):

    def __init__(self, input_size, output_size):
        self.bias = np.random.rand(1, output_size) - 0.5
        self.weights = np.random.rand(input_size, output_size) - 0.5


    # computes the output of a layer
    def forward_propagation(self, X):
        self.input = X
        self.output = np.dot(X, self.weights) + self.bias
        return self.output

    # re-computes or "learns" the new parameters
    def backward_propagation(self, out_gradient, alpha):
        weight_gradient = np.dot(self.input.T, out_gradient)
        bias_gradient = out_gradient


        self.weights -= alpha * weight_gradient
        self.bias -= alpha * bias_gradient

        return np.dot(out_gradient, self.weights.T)


