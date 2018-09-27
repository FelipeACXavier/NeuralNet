import numpy as np

class NeuralNet:
    # Initiaite the neural net
    def __init__(self, inputs, outputs, layers, nodes):
        self.inputs = inputs
        self.outputs = outputs
        self.layers = layers
        self.nodes = nodes
        self.learningRate = 0.5

        self.inputWeights = self.setWeights(1, len(self.inputs), self.nodes)
        self.hiddenWeights = self.setWeights(self.layers, self.nodes, self.nodes)
        self.outputWeights = self.setWeights(1, self.nodes, len(self.outputs))
        
        self.hiddenBias = self.setBias(self.layers, self.nodes)
        self.outputBias = self.setBias(1, self.nodes)

    # Returns a list of numpy arrays with random weight for each layer  
    def setWeights(self, layers, coming, going):
        temp = []
        for i in range(layers):
            temp.append(np.random.rand(coming, going))
        return temp

    # Returns a list with bias for every node in each layer
    def setBias(self, layers, nodes):
        temp = []
        for i in range(layers):
            temp.append([])
            for j in range(nodes):
                temp[i].append(np.random.uniform())
        return temp
  
    # Starts with a rate of 0.5 but it can be changed using this function
    def setLearnRate(self, rate):
        self.learningRate = rate
    
    # Definition of sigmoid function to keep outputs between 0 and 1
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

n = NeuralNet([1, 2], [1], 2, 3)
#print("Input: " + str(n.inputWeights))
#print("Hidden: " + str(n.hiddenWeights))
#print("Ouput: " + str(n.outputWeights))