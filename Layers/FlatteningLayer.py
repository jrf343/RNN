from Layer import Layer
import numpy as np

class FlatteningLayer(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, dataIn):
        self.input_shape = dataIn.shape
        num_images = dataIn.shape[0]
        flattened_images = dataIn.reshape(num_images, -1, order='F')  # Use column-major ordering
        return flattened_images
    
    def gradient(self):
        pass
    
    def backward(self, gradIn):
        return gradIn.reshape(self.input_shape, order='F')  # Reshape back with column-major ordering