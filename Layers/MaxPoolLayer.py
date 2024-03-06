from Layer import Layer
import numpy as np

class MaxPoolLayer(Layer):
    def __init__(self, width=4, stride=4):
        super().__init__()
        self.width = width
        self.stride = stride
        self.indices = None
    
    def forward(self, dataIn):
        num_images, input_rows, input_cols = dataIn.shape
        pool_width = self.width
        stride = self.stride
        
        output_rows = (input_rows - pool_width) // stride + 1
        output_cols = (input_cols - pool_width) // stride + 1
        
        output = np.zeros((num_images, output_rows, output_cols))
        self.indices = np.zeros_like(dataIn, dtype=int)

        for n in range(num_images):
            for i in range(0, input_rows - pool_width + 1, stride):
                for j in range(0, input_cols - pool_width + 1, stride):
                    patch = dataIn[n, i:i+pool_width, j:j+pool_width]
                    max_value = np.max(patch)
                    max_index = np.unravel_index(np.argmax(patch), patch.shape)
                    output[n, i//stride, j//stride] = max_value
                    self.indices[n, i + max_index[0], j + max_index[1]] = 1
                    
        self.setPrevIn(dataIn)
        self.setPrevOut(output)

        return output
    
    def gradient(self):
        pass
    
    def backward(self, gradIn):
        num_images, output_rows, output_cols = gradIn.shape
        pool_width = self.width
        stride = self.stride
        
        gradOut = np.zeros_like(self.indices, dtype=gradIn.dtype)  # Initialize gradOut with the same dtype as gradIn
        
        for n in range(num_images):
            for i in range(output_rows):
                for j in range(output_cols):
                    grad_value = gradIn[n, i, j]
                    indices_row, indices_col = np.where(self.indices[n, i*stride:i*stride+pool_width, j*stride:j*stride+pool_width] == 1)
                    gradOut[n, indices_row + i*stride, indices_col + j*stride] = grad_value
        
        return gradOut