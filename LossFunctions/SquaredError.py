import numpy as np
np.random.seed(0)

class SquaredError():
    # Input: Y is an N by K matrix of target values.
    # Input: Yhat is an N by K matrix of estimated values.  
    # Where N can be any integer >= 1
    # Output:  A single floating point value.
    def eval(self,Y, Yhat):
        return np.mean((Y - Yhat)*(Y - Yhat))

    # Input: Y is an N by K matrix of target values.
    # Input: Yhat is an N by K matrix of estimated values.
    # Output:  An N by K matrix.
    def gradient(self,Y, Yhat):
        return -2 * (Y - Yhat)

