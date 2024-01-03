import numpy as np

#activations functions will not be called here
class Dense():

    def __init__(self, input_dim, output_dim):
        
        # Generate a Gaussian randomly generated weights matrix of size input_dim x output_dim
        gaussianRng = np.random.Generator.standard_normal()
        self.weights = gaussianRng(size=(input_dim, output_dim))

        # Generate a Gaussian randomly generated bias column vector
        self.bias = gaussianRng(size=(input_dim, 1))



    def forward(self, input):

        # Forward = weights . input + bias (matrix multiplication)
        return np.dot(self.weights, input) + self.bias



    def backward(self, output_gradient, learning_rate):
        pass