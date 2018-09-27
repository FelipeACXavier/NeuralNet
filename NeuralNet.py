import numpy as np

class NeuralNet:
    # Initiaite the neural net
    def __init__(self, inputs, outputs, layers, nodes):
        self.inputs = inputs # int
        self.outputs = outputs # int
        self.layers = layers # int
        self.nodes = nodes   # int
        self.hidden_calc = [] # list
        self.learningRate = 0.5

        self.inputWeights = self.setWeights(2, self.inputs, self.nodes) # list of np.array
        self.hiddenWeights = self.setWeights(self.layers, self.nodes, self.nodes) # list of np.array
        self.outputWeights = self.setWeights(2, self.nodes, self.outputs) # list of np.array
        
        self.hiddenBias = self.setBias(self.layers, self.nodes) # list of np.array
        self.outputBias = self.setBias(1, self.outputs) # list of np.array

    # Returns a list of numpy arrays with random weight for each layer  
    def setWeights(self, layers, coming, going):
        temp = []
        for i in range(layers - 1):
            temp.append(np.random.rand(going, coming))
        return temp

    # Returns a list with bias for every node in each layer
    def setBias(self, layers, nodes):
        temp = []
        for i in range(layers):
            temp.append([])
            for j in range(nodes):
                temp[i].append(np.random.uniform(-1,1))
        return temp
  
    # Starts with a rate of 0.5 but it can be changed using this function
    def setLearningRate(self, rate):
        self.learningRate = rate
    
    # Definition of sigmoid function to keep outputs between 0 and 1
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def feedForward(self, input): 
        #print(input)
        for i in range(self.layers + 1):
            if i is 0:
                self.hidden_calc.append(np.dot(self.inputWeights[0], input))
                self.hidden_calc[i] = self.sigmoid(np.add(self.hidden_calc[i], self.hiddenBias[0])) 
            elif i is self.layers:
                self.hidden_calc.append(np.dot(self.outputWeights[0], self.hidden_calc[i - 1]))
                self.hidden_calc[i] = self.sigmoid(np.add(self.hidden_calc[i], self.outputBias)) 
            else:
                self.hidden_calc.append(np.dot(self.hiddenWeights[i - 1], self.hidden_calc[i - 1]))
                self.hidden_calc[i] = self.sigmoid(np.add(self.hidden_calc[i - 1], self.hiddenBias[i]))
            print("Current node: " + str(self.hidden_calc[i]))
        return self.hidden_calc[self.layers]

    def train(self, inputs, target):
        self.out = self.feedForward(inputs)

        self.output_error = np.subtract(self.out, target)
        print(self.out)
        print(self.output_error)

n = NeuralNet(2, 1, 1, 2)
#print("Input: " + str(n.inputWeights))
#print("Hidden: " + str(n.hiddenWeights))
#print("Ouput: " + str(n.outputWeights))
#print(n.outputBias)
#print("Bias: " + str(n.hiddenBias))
#print("train data: " + str(n.feedForward([1,2])))
n.train([1, 0], [1])