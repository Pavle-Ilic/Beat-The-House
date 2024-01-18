from ReLu import relu, reluPrime
from MSE import mse, msePrime
from Linear import linear, linearPrime
from DenseLayerClass import Dense
import numpy as np
import copy

class Network():
    def __init__(self, learning_rate):
        self.network = []
        self.learning_rate = learning_rate

    def add(self, layer):
        if self.network:
            prev_layer = self.network[-1]
            #get dimension of previous layer and new layer
            prev_layer_output_dim = prev_layer.weights.shape[0]
            new_layer_input_dim = layer.weights.shape[1]

            #checking if dimensions match
            if prev_layer_output_dim != new_layer_input_dim:
                raise ValueError("Input dimension of the new layer must match the output dimension of the previous layer")

        self.network.append(layer)
        
    #make a deep copy of one network to another
    def copyNetwork(self):
        copied_network = Network(self.learning_rate)
        for layer in self.network:
            copied_layer = copy.deepcopy(layer)
            copied_network.add(copied_layer) 
        return copied_network

    def predict(self, input):
        output = input
        for i in range(len(self.network) - 1):
            output = relu(self.network[i].forward(output))
        output = linear(self.network[-1].forward(output))
        return output

    def fit(self, episodes, x_data, y_data, loss):
        for count in range(episodes):
            error = 0
            for x, y in zip(x_data, y_data):
                output = self.predict(x) # forward prop

                error += mse(y, output) # error, wip

                grad = msePrime(y, output) # backward prop
                for layer in reversed(self): # wip
                    grad = layer.backward(grad, self.learning_rate)
            
            error /= len(x_data) # divde error by the length of x

    def save(self):
        pass

    def load(self):
        pass
