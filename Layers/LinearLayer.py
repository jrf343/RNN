from .Layer import Layer
import numpy as np
np.random.seed(0)

class LinearLayer(Layer):
    # Input: None
    # Output: None
    def __init__(self):
        super().__init__()
    
    # Input: dataIn, an NxD matrix 
    # Output: an NxD matrix 
    def forward(self, dataIn):
        # g(z) = z
        # self.setPrevIn(dataIn)
        # self.setPrevOut(dataIn)

        ## ADDED FOR RNN FUNCTIONALITY ##
        output = dataIn
        self.addPrevIn(dataIn)
        self.addPrevOut(output)
        
        return output
    
    # Input: None
    # Output: Either an NxD matrix or an Nx(DxD) tensor
    def gradient(self, t_inp):
        # The gradient is constructed by encoding the diagonal values in a row
        # The gradient for a linear layer is the identity matrix

        # Create a vector of ones for each observation
        return np.ones(self.getPrevOut()[t_inp].shape)

    # Input: The backcoming gradient
    # Output: The updated gradient to be backpropagated
    def backward(self, gradIn, t_inp):
        # The backward gradient is the element-wise (Hadamard) product of the incoming gradient and the layer's gradient
        return gradIn * self.gradient(t_inp)
