from .Layer import Layer
import numpy as np
np.random.seed(0) # For reproducibility

class FullyConnectedLayer(Layer):
    # Input: sizeIn, the number of features of data coming in
    # Input: sizeOut, the number of features for the data coming out
    # Output: None
    def __init__ (self, sizeIn, sizeOut, random=True, randstate=0):
        super().__init__()
        if(random):
            self.weights = np.random.uniform(-1e-4, 1e-4, size=(sizeIn, sizeOut))
            self.biases = np.random.uniform(-1e-4, 1e-4, size=(1, sizeOut))
        else:
            self.weights = np.random.RandomState(randstate).uniform(-1e-4, 1e-4, size=(sizeIn, sizeOut))
            self.biases = np.random.RandomState(randstate+10).uniform(-1e-4, 1e-4, size=(1, sizeOut))

        ## ADDED FOR RNN FUNCTIONALITY ##
        self.weights_grad_accum = np.zeros((sizeIn, sizeOut))
        self.biases_grad_accum = np.zeros((1, sizeOut))
    
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
        
        # self.setPrevIn(dataIn)
        # self.setPrevOut(output)

        ## ADDED FOR RNN FUNCTIONALITY ##
        self.addPrevIn(dataIn)
        self.addPrevOut(output)
        
        return output
    
    ## ADDED FOR RNN FUNCTIONALITY ##
    def forward_with_feedback(self, dataIn, feedback):
        # y = xW + b
        output = (np.dot(dataIn, self.weights) + self.biases) + feedback
        
        self.addPrevIn(dataIn)
        self.addPrevOut(output)
        
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
    
    ## ADDED FOR RNN FUNCTIONALITY ##
    def updateWeightsGradAccum(self, gradIn, t_inp):
        self.weights_grad_accum += (self.getPrevIn()[t_inp].T @ gradIn)/gradIn.shape[0]

    def updateBiasesGradAccum(self, gradIn):
        self.biases_grad_accum += np.sum(gradIn, axis=0)/gradIn.shape[0]

    
    # Input: The backcoming gradient
    # Output: None
    def updateWeights(self, gradIn, eta = 1e-4):
        # dJdb = np.sum(gradIn, axis = 0)/gradIn.shape[0]
        # dJdW = (self.getPrevIn().T @ gradIn)/gradIn.shape[0]

        dJdW = self.weights_grad_accum
        dJdb = self.biases_grad_accum
        
        newWeights = self.getWeights() - (eta * dJdW)
        newBiases = self.getBiases() - (eta * dJdb)
        
        self.setWeights(newWeights)
        self.setBiases(newBiases)