from ReLu import relu, reluPrime
from MSE import mse, msePrime
from Linear import linear, linearPrime
import numpy as np

class Network():
    def __init__(self, learning_rate):
        self.network = []

    def add(self, layer):
        pass
    
    #make a deep copy of one network to another
    def copyNetwork(self):
        pass

    def predict(self, input):
        output = input
        for i in range(len(self.network) - 1):
            output = relu(self.network[i].forward(output))
        output = linear(self.network[-1].forward(output))
        return output

    def fit(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

