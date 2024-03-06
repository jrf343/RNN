from Layer import Layer
import numpy as np

class TanhLayer(Layer):
    # Input: None
    # Output: None
    def __init__(self):
        super().__init__()
    
    # Input: dataIn, an NxD matrix 
    # Output: an NxD matrix 
    def forward(self, dataIn):
        # g(z) = (e^(z) - e^(-z)) / (e^(z) + e^(-z))
        output = np.tanh(dataIn)
        
        self.setPrevIn(dataIn)
        self.setPrevOut(output)
        
        return output
    
    # Input: None
    # Output: Either an NxD matrix or an Nx(DxD) tensor
    def gradient(self):
        # The gradient of the hyperbolic tangent function is g'(z) = 1 - g(z)^2
        tanh_output = self.getPrevOut()
        return 1 - tanh_output**2
 
    # Input: The backcoming gradient
    # Output: The updated gradient to be backpropagated   
    def backward(self, gradIn):
        # The backward gradient is the element-wise (Hadamard) product of the incoming gradient and the layer's gradient
        return gradIn * self.gradient()