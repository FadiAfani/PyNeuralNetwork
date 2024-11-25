from layer import Layer
import numpy as np

class Activation(Layer):

    def __init__(self, function):
        if function == "sigmoid":
            self.activation = self.sigmoid
            self.activation_der = self.sigmoid_der
        
    # ReLu activation function 
    def relu(self, X: np.ndarray):
        zero_tensor = np.zeros(X.shape)
        return np.maximum(zero_tensor, X)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_der(self, x):
        res = self.sigmoid(x)
        return res * (1 - res)
    
    def forward_propagation(self, X):
        self.input = X
        self.output = self.activation(x=X)
        return self.output

    def backward_propagation(self, out_gradient, alpha):
        return np.multiply(out_gradient, self.activation_der(x=self.input))