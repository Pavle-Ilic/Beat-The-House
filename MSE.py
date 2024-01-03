import numpy as np

# Mean squared error desired output versus the actual output of the network
def mse(y_actual, y_prediction, derivativeFlag=0):
    
    # If we are not using the derivate for Y, we can use the standard MSE formula
    if(derivativeFlag == 0):
        e = np.mean(np.power(y_actual-y_prediction),2)
        return e
    
    # If we are using the derivative for Y, we need to take the derivative of the MSE formula with respect to Y 
    ePrime = 2 * (y_prediction-y_actual) / np.size(y_actual)
    return ePrime

        


# This method is implemented in mse (when derivativeFlag = 1)
def msePrime():
    pass