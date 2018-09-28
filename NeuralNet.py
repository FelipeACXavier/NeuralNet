import numpy as np

class NeuralNet:
    # Initiaite the neural net
    def __init__(self, inputs, outputs, layers, nodes):
        self.inputs = inputs # int
        self.outputs = outputs # int
        self.layers = layers + inputs + outputs # int
        self.nodes = nodes   # int
        self.learningRate = 0.5
        self.weights = 
        self.bias = self.() 
        

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

    # Derivative of sigmoid function
    def dsigmoid(self, x):
        return x*(1-x)
    
    # Feed the input from layer to layer until the output
    def feedForward(self, input): 
        return

    def train(self, input, target):
        return

# ---------------------------------------------------------
# ---------------------------------------------------------
# ---------------------------------------------------------
n = NeuralNet(2, 1, 1, 2)
#print("Input: " + str(n.inputWeights))
print("Hidden: " + str(n.hiddenWeights))
#print("Ouput: " + str(n.outputWeights))
#print(n.outputBias)
#print("Bias: " + str(n.hiddenBias))
#print("train data: " + str(n.feedForward([1,2])))
der = [{"input":[0,0], "target":[0]},
        {"input":[0,1], "target":[1]},
        {"input":[1,0], "target":[1]},
        {"input":[1,1], "target":[0]}]
        
for i in range(1000):
    a = np.random.randint(0,4)
    # print(der[a]["input"] + der[a]["target"])
    n.train(der[a]["input"], der[a]["target"])

print(n.feedForward([0,0]))
print(n.feedForward([0,1]))
print(n.feedForward([1,0]))
print(n.feedForward([1,1]))

#print("Input: " + str(n.inputWeights))
#print("Hidden: " + str(n.hiddenWeights))
#print("Ouput: " + str(n.outputWeights))