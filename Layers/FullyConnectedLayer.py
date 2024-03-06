from Layer import Layer
import numpy as np

class FullyConnectedLayer(Layer):
    # Input: sizeIn, the number of features of data coming in
    # Input: sizeOut, the number of features for the data coming out
    # Output: None
    def __init__ (self, sizeIn, sizeOut):
        super().__init__()
        self.weights = np.random.uniform(-1e-4, 1e-4, size=(sizeIn, sizeOut))
        self.biases = np.random.uniform(-1e-4, 1e-4, size=(sizeOut, 1))
    
    # Input: None
    # Output: The sizeIn x sizeOut weight matrix
    def getWeights(self): 
        return self.weights
   
    # Input: The sizeIn x sizeOut weight matrix
    # Output: None
    def setWeights(self, weights):
        self.weights = weights
    
    # Input: None
    # Ouput: The 1 x sizeOut bias vector
    def getBiases(self):
        return self.biases
    
    # Input: The 1 x sizeOut bias vector
    # Ouput: None
    def setBiases(self, biases):
        self.biases = biases
    
    # Input: dataIn, an NxD matrix 
    # Output: an NxD matrix 
    def forward(self, dataIn):
        # y = xW + b
        output = np.dot(dataIn, self.weights) + self.biases
        
        self.setPrevIn(dataIn)
        self.setPrevOut(output)
        
        return output
    
    # Input: None
    # Output: Either an NxD matrix or an Nx(DxD) tensor
    def gradient(self):
        return np.transpose(self.getWeights())
    
    # Input: The backcoming gradient
    # Output: The updated gradient to be backpropagated
    def backward(self, gradIn):
        # Compute the gradient of the loss with respect to the input of the layer
        gradOut = gradIn @ self.gradient()
        return gradOut
    
    # Input: The backcoming gradient
    # Output: None
    def updateWeights(self, gradIn, eta = 1e-4):
        dJdb = np.sum(gradIn, axis = 0)/gradIn.shape[0]
        dJdW = (self.getPrevIn().T @ gradIn)/gradIn.shape[0]
        
        newWeights = self.getWeights() - (eta * dJdW)
        newBiases = self.getBiases() - (eta * dJdb)
        
        self.setWeights(newWeights)
        self.setBiases(newBiases)