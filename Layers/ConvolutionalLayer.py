from Layer import Layer
import numpy as np

class ConvolutionalLayer(Layer):
    # Input: kernelWidth
    # Input: kernelHeight
    # Output: None
    def __init__ (self, kernelWidth, kernelHeight):
        super().__init__()
        self.weights = np.random.uniform(-1e-4, 1e-4, size=(kernelWidth, kernelHeight))
    
    # Input: None
    # Output: The weight matrix
    def getWeights(self): 
        return self.weights
   
    # Input: The weight matrix
    # Output: None
    def setWeights(self, weights):
        self.weights = weights
    
    # Input: dataIn, an NxD matrix 
    # Output: an NxD matrix 
    def forward(self, dataIn):
        # 2D cross-correlation
        output = self.crossCorrelate2D(dataIn, self.getWeights())
        
        self.setPrevIn(dataIn)
        self.setPrevOut(output)
        
        return output
    
    def gradient(self):
        pass
    
    def backward(self, gradIn):
        pass
    
    # Input: The backcoming gradient
    # Output: None
    def updateWeights(self, gradIn, eta=1e-4):
        dataIn = self.getPrevIn()
        num_images, outputRows, outputCols = gradIn.shape
        kernelRows, kernelCols = self.getWeights().shape
        gradWeights = np.zeros_like(self.getWeights(), dtype=gradIn.dtype)

        for n in range(num_images):
            for i in range(outputRows):
                for j in range(outputCols):
                    gradWeights += gradIn[n, i, j] * dataIn[n, i:i+kernelRows, j:j+kernelCols]

        # Update weights
        new_weights = self.getWeights() - (eta * gradWeights)
        self.setWeights(new_weights)

        return gradWeights
        
    def crossCorrelate2D(self, dataIn, kernel):
        num_images, inRows, inCols = dataIn.shape
        kernelRows, kernelCols = kernel.shape
        
        outputRows = inRows - kernelRows + 1
        outputCols = inCols - kernelCols + 1
        
        output = np.zeros((num_images, outputRows, outputCols))
        
        for n in range(num_images):
            for i in range(outputRows):
                for j in range(outputCols):
                    output[n, i, j] = np.sum(kernel * dataIn[n, i:i+kernelRows, j:j+kernelCols])
        
        return output