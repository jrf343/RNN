import numpy as np

class LogLoss():
    # Input: Y is an N by K matrix of target values.
    # Input: Yhat is an N by K matrix of estimated values.  
    # Where N can be any integer >= 1
    # Output:  A single floating point value.
    def eval(self,Y, Yhat):
        epsilon = 1e-15
        return -np.mean((Y * (np.log(Yhat + epsilon)) + ((1 - Y) * (np.log(1 - Yhat + epsilon)))))
        

    # Input: Y is an N by K matrix of target values.
    # Input: Yhat is an N by K matrix of estimated values.
    # Output:  An N by K matrix.
    def gradient(self,Y, Yhat):
        epsilon = 1e-15
        return -(Y - Yhat)/((Yhat * (1 - Yhat)) + epsilon)

