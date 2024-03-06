from Layer import Layer
import numpy as np

class LogisticSigmoidLayer(Layer):
    # Input: None
    # Output: None
    def __init__(self):
        super().__init__()
    
    # Input: dataIn, an NxD matrix 
    # Output: an NxD matrix 
    def forward(self, dataIn):
        # g(z) = 1/(1+e^(-z))
        output = 1/(1 + np.exp(-dataIn))

        self.setPrevIn(dataIn)
        self.setPrevOut(output)
        
        return output
    
    # Input: None
    # Output: Either an NxD matrix or an Nx(DxD) tensor
    def gradient(self):
        # The gradient of the logistic sigmoid function is g'(z) = g(z) * (1 - g(z))
        sigmoid_output = self.getPrevOut()
        return sigmoid_output * (1 - sigmoid_output)
 
    # Input: The backcoming gradient
    # Output: The updated gradient to be backpropagated   
    def backward(self, gradIn):
        # The backward gradient is the element-wise (Hadamard) product of the incoming gradient and the layer's gradient
        return gradIn * self.gradient()