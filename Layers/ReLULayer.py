from .Layer import Layer
import numpy as np

class ReLULayer(Layer):
    # Input: None
    # Output: None
    
    def __init__(self):
        super().__init__()
    
    # Input: dataIn, an NxD matrix 
    # Output: an NxD matrix 
    def forward(self, dataIn):
        # g(z) = max(0,z)
        output = np.maximum(0, dataIn)
        
        # self.setPrevIn(dataIn)
        # self.setPrevOut(output)

        ## ADDED FOR RNN FUNCTIONALITY ##
        self.addPrevIn(dataIn)
        self.addPrevOut(output)
        
        return output
    
    # Input: None
    # Output: Either an NxD matrix or an Nx(DxD) tensor
    def gradient(self, t_inp):
        # The gradient of ReLU function is 1 for positive values and 0 for negative values
        return np.where(self.getPrevOut()[t_inp] > 0, 1, 0)

    # Input: The backcoming gradient
    # Output: The updated gradient to be backpropagated    
    def backward(self, gradIn, t_inp):
        # The backward gradient is the element-wise (Hadamard) product of the incoming gradient and the layer's gradient
        return gradIn * self.gradient(t_inp)