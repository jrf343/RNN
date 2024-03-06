from Layer import Layer
import numpy as np

class InputLayer(Layer):
    # Input: dataIn, an NxD matrix 
    # Output: None
    def __init__(self, dataIn):
        super().__init__()
        self.meanX = np.mean(dataIn, axis=0)
        self.stdX = np.std(dataIn, axis=0)
        
        self.stdX[self.stdX == 0] = 1
        
    # Input: dataIn, an NxD matrix 
    # Output: an NxD matrix 
    def forward(self, dataIn):
        zscored_data = (dataIn - self.meanX) / self.stdX
        
        self.setPrevIn(dataIn)
        self.setPrevOut(zscored_data)
        
        return zscored_data
        
    def gradient(self):
        pass
    
    def backward(self, gradIn):
        pass
