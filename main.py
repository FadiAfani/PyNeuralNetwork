from hidden import Hidden
from network import Network
from activation import Activation
import numpy as np


neural_network = Network()

X_train, y_train, X_test, y_test = neural_network.preprocess()

# add layers to the network
#neural_network.add_layer(Hidden(X_train.shape[1], 50))
#neural_network.add_layer(Activation("sigmoid"))
#neural_network.add_layer(Hidden(50, 10))
#neural_network.add_layer(Activation("sigmoid"))

# training
#loss_history = neural_network.mini_batch_gradient_descent(X=X_train, y=y_train)


# save model
#neural_network.save_model("model")

# load model 
neural_network.load_model("./models/model.json")

# model evaluation - testing set
model_accuracy_train = neural_network.compute_accuracy(X=X_train, y=y_train)
model_accuracy_test = neural_network.compute_accuracy(X=X_test, y=y_test)


print(f'training_accuracy: {model_accuracy_train}, testing_accuracy: {model_accuracy_test}')


