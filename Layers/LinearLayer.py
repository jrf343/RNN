from Layer import Layer
import numpy as np

class LinearLayer(Layer):
    # Input: None
    # Output: None
    def __init__(self):
        super().__init__()
    
    # Input: dataIn, an NxD matrix 
    # Output: an NxD matrix 
    def forward(self, dataIn):
        # g(z) = z
        self.setPrevIn(dataIn)
        self.setPrevOut(dataIn)
        
        return dataIn
    
    # Input: None
    # Output: Either an NxD matrix or an Nx(DxD) tensor
    def gradient(self):
        # The gradient is constructed by encoding the diagonal values in a row
        # The gradient for a linear layer is the identity matrix

        # Create a vector of ones for each observation
        return np.ones(self.getPrevOut().shape)

    # Input: The backcoming gradient
    # Output: The updated gradient to be backpropagated
    def backward(self, gradIn):
        # The backward gradient is the element-wise (Hadamard) product of the incoming gradient and the layer's gradient
        return gradIn * self.gradient()
