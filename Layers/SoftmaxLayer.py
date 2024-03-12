from .Layer import Layer
import numpy as np

class SoftmaxLayer(Layer):
    # Input: None
    # Output: None
    def __init__(self):
        super().__init__()
    
    # Input: dataIn, an NxD matrix 
    # Output: an NxD matrix 
    def forward(self, dataIn):
        # g(z) = e^(z - max(z))/(sum_(i)e^(z_(i) - max(z)))
        if dataIn.ndim == 1:
            # If 1D, make it a row vector (2D)
            dataIn = dataIn.reshape(1, -1)
        
        exp_values = np.exp(dataIn - np.max(dataIn, axis=1, keepdims=True))
        output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.setPrevIn(dataIn)
        self.setPrevOut(output)
        
        return output
    
    # Input: None
    # Output: Either an NxD matrix or an Nx(DxD) tensor
    def gradient(self):
        softmax_output = self.getPrevOut()
        n_classes = softmax_output.shape[1]
        jac_matrix = np.zeros((len(softmax_output), n_classes, n_classes))

        for i in range(len(softmax_output)):
            g_z_i = np.diag(softmax_output[i]) - np.outer(softmax_output[i], softmax_output[i])
            jac_matrix[i] = g_z_i

        return jac_matrix
 
    # Input: The backcoming gradient
    # Output: The updated gradient to be backpropagated   
    def backward(self, gradIn):
        # The backward gradient involves the tensor product with einsum
        return np.einsum('...i,...ij', gradIn, self.gradient())