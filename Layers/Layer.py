from abc import ABC, abstractmethod

###BASE CLASS###
class Layer(ABC):
    def __init__(self):
        self.__prevIn = []
        self.__prevOut = []
    
    def setPrevIn(self,dataIn):
        self.__prevIn = dataIn
    
    def setPrevOut(self, dataOut):
        self.__prevOut = dataOut
    
    def getPrevIn(self):
        return self.__prevIn
    
    def getPrevOut(self):
        return self.__prevOut
    
    #-----ADDED FOR RNN FUNCTIONALITY-----#
    def addPrevIn(self, dataIn):
        self.__prevIn.append(dataIn)
    
    def accessPrevIn(self, index):
        return self.__prevIn[index]
    
    def addPrevOut(self, dataOut):
        self.__prevOut.append(dataOut)

    def accessPrevOut(self, index):
        return self.__prevOut[index]
    #-----ADDED FOR RNN FUNCTIONALITY-----#
    
    @abstractmethod
    def forward(self,dataIn):
        pass
    
    @abstractmethod
    def gradient(self):
        pass
    
    @abstractmethod
    def backward(self,gradIn):
        pass
        