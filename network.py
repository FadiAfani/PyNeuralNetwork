from layer import Layer
from hidden import Hidden
from activation import Activation
import numpy as np
import pandas as pd
import json



class Network:

    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def preprocess(self):
        data_train = np.array(pd.read_csv('./datasets/mnist_train.csv'))
        data_test = np.array(pd.read_csv('./datasets/mnist_test.csv'))
        np.random.shuffle(data_train)
        np.random.shuffle(data_test)
        X_train = data_train[:, 1:]
        y_train = data_train.T[0]
        X_test = data_test[:, 1:]
        y_test = data_test.T[0]

        X_train = (1/255) * X_train
        X_test = (1/255) * X_test

        return X_train, y_train, X_test, y_test

    def error_gradient(self, y, y_hat):
        m = len(y)
        return (2/m) * (y - y_hat)

    def compute_cost(self, x, y, network_output):
        y_vector = [1 if i == y else 0 for i in range(10)]
        cost = (network_output - y_vector)**2
        error_gradient = self.error_gradient(network_output, y_vector)
        return cost, error_gradient

    def create_batches(self, X, y, batch_size):
        dataset = np.c_[y, X]
        np.random.shuffle(dataset)
        batch_list = []
        num_batches = X.shape[0] // batch_size
        for _ in range(num_batches):
            # pick a batch (no replacement)
            batch = dataset[np.random.choice(dataset.shape[0], batch_size, replace=False), :]
            batch_list.append(batch)
        return batch_list

    def mini_batch_gradient_descent(self, X, y, batch_size=128, eps=0.01, alpha=0.1, epochs=50):

        loss_history = []
        m = len(y)
        batches = self.create_batches(X, y, batch_size)

        for e in range(epochs):
            # pick a batch
            batches_cost = []
            for batch in batches:
                cost_list = []
                #error_gradient_list = []
                # iterate over the training examples
                X_batch = batch[:, 1:]
                y_batch = batch.T[0]
                for i in range(batch_size):
                    # forward propagation -> compute the network's output
                    X_np_array = np.array([X_batch[i]])
                    network_output = X_np_array.copy()
                    for layer in self.layers:
                        network_output = layer.forward_propagation(network_output)
                    cost, error_gradient = self.compute_cost(X_np_array, y_batch[i], network_output)
                    #error_gradient_list.append(error_gradient)
                    cost_list.append(cost)
                    # backward propagation -> fine tune the parameters
                    for layer in self.layers[::-1]:
                        error_gradient = layer.backward_propagation(error_gradient, alpha)
                batches_cost.append((1 / batch_size) * np.sum(cost_list))
            loss_history.append(np.mean(batches_cost))
            if loss_history[-1] <= eps:
                break

            if e % 10 == 0:
                print("Epoch:", e, "Loss:", loss_history[e])
            

        return loss_history

    # predict an instance given a trained network
    def predict(self, x):
        y_pred = x.copy()
        for layer in self.layers:
            y_pred = layer.forward_propagation(y_pred)
        return np.argmax(y_pred, axis=1)

    def compute_accuracy(self, X, y):
        correct_predictions = 0
        predictions = self.predict(X)
        m = len(y)
        for i in range(m):
            if predictions[i] == y[i]:
                correct_predictions += 1
        return correct_predictions / m
    
    def save_model(self, name):
        model = {}
        for i, layer in enumerate(self.layers):
            layer_type = None
            if isinstance(layer, Hidden):
                model[f'layer_{i}'] = {
                    "type": "hidden",
                    "bias": layer.bias.tolist(),
                    "weights": layer.weights.tolist()
                }
            else:
                model[f'layer_{i}'] = {
                    "type": "activation",
                    "function": "sigmoid"
                }

        with open(f'./models/{name}.json', "w") as f:
            json.dump(model, f, indent=4)

    def load_model(self, path):
        self.layers = []
        try:
            with open(path, "r") as f:
                model = json.load(f)
                for layer in model.values():
                    if layer['type'] == 'activation':
                        self.add_layer(Activation(layer['function']))
                    else:
                        np_bias = np.array(layer['bias'])
                        np_weights = np.array(layer['weights'])
                        hidden = Hidden(np_weights.shape[0], np_weights.shape[1])
                        hidden.bias = np_bias
                        hidden.weights = np_weights
                        self.add_layer(hidden)
        except FileNotFoundError as e:
            print("no model found !")

        



