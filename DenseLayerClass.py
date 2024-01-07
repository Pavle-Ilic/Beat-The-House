import numpy as np

#activations functions will not be called here
class Dense():

    def __init__(self, input_dim, output_dim):
        
        # Generate a Gaussian randomly generated weights matrix of size input_dim x output_dim
        gaussianRng = np.random.Generator.standard_normal()
        self.weights = gaussianRng(size=(input_dim, output_dim))

        # Generate a Gaussian randomly generated bias column vector
        gaussianRng = np.random.Generator.standard_normal()
        self.bias = gaussianRng(size=(input_dim, 1))



    def forward(self, input):

        self.input = input

        # Forward = weights . input + bias (matrix multiplication)
        return np.dot(self.weights, self.input) + self.bias



    def backward(self, output_gradient, learning_rate):

        # Calculate the derivative of the error wrt the weights
        # dE/dW = dE/dY dot X^t (X transposed)
        self.weights_gradient = np.dot(output_gradient,self.input.T)

        # update the weights and bias based off the weights gradient
        self.weights -= learning_rate * self.weights_gradient
        self.bias -= learning_rate * output_gradient   # note that dE/dB = dE/dY

        # return the derivative of the error wrt the input
        # dE/dX = W^t dot dE/dY
        self.input_gradient = np.dot(self.weights.T, output_gradient)
        return self.input_gradient



