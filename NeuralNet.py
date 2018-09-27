import numpy as np

class NeuralNet:
    # Initiaite the neural net
    def __init__(self, inputs, outputs, layers, nodes):
        self.inputs = inputs # int
        self.outputs = outputs # int
        self.layers = layers # int
        self.nodes = nodes   # int
        # Initialize lists with ones
        self.hidden_calc = [] # list
        self.hiddenErrors = [] # list
        for i in range(self.layers + 1): # hidden_calc
                self.hidden_calc.append(1)
        for i in range(self.layers): # hiddenErrors
            self.hiddenErrors.append(1)

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

    def dsigmoid(self, x):
        return x*(1-x)
    
    # Feed the input from layer to layer until the output
    def feedForward(self, input): 
        self.imp = input
        for i in range(self.layers + 1):
            if i is 0:
                self.hidden_calc[i] = np.dot(self.inputWeights[0], input)
                self.hidden_calc[i] = self.sigmoid(np.add(self.hidden_calc[i], self.hiddenBias[0]))
            elif i is self.layers:
                self.hidden_calc[i] = np.dot(self.outputWeights[0], self.hidden_calc[i - 1])
                self.hidden_calc[i] = self.sigmoid(np.add(self.hidden_calc[i], self.outputBias)) 
            else:
                self.hidden_calc[i] = np.dot(self.hiddenWeights[i - 1], self.hidden_calc[i - 1])
                self.hidden_calc[i] = self.sigmoid(np.add(self.hidden_calc[i - 1], self.hiddenBias[i]))
        # print("Hidden_Calc: " + str(self.hidden_calc))
        return self.hidden_calc[self.layers]

    def train(self, input, target):
        # Output error
        self.out = self.feedForward(input)
        self.outputErrors = np.subtract(self.out, target)
        self.delta = self.dsigmoid(self.outputErrors)
        print("output: " + str(self.out))
        print("error: " + str(self.outputErrors))
    
    def update(self):
        # Update output weights
        a = self.learningRate * np.multiply(self.hidden_calc[self.layers], self.hiddenErrors[0].T)
        #b = self.hidden_calc[self.layers - 1][np.newaxis]
        self.outputWeights += a#np.dot(a.T, b)
        
        # Update hidden weights
        for i in range(self.layers - 1):
            a = self.learningRate * np.dot(self.hidden_calc[self.layers-1-i], self.hiddenErrors[i+1])
            #b = self.hidden_calc[self.layers - 1 - i]
            self.hiddenWeights[i] += a#np.dot(a.T, b)
        
        # Update input weights
        a = self.learningRate * np.dot(self.imp, self.hiddenErrors[self.layers-1])
        #self.imp[np.newaxis]
        self.inputWeights += a#np.dot(a.T, self.imp)

    def Error(self, weight, which):
        for i in range(len(weight)):
            denominator += weight[i]
        return weight[which]/denominator

# ---------------------------------------------------------
# ---------------------------------------------------------
# ---------------------------------------------------------
n = NeuralNet(2, 1, 1, 2)
#print("Input: " + str(n.inputWeights))
#print("Hidden: " + str(n.hiddenWeights))
#print("Ouput: " + str(n.outputWeights))
#print(n.outputBias)
#print("Bias: " + str(n.hiddenBias))
#print("train data: " + str(n.feedForward([1,2])))
der = [{"input":[0,0], "target":[0]},
        {"input":[0,1], "target":[1]},
        {"input":[1,0], "target":[1]},
        {"input":[1,1], "target":[0]}]
        
for i in range(10):
    a = np.random.randint(0,4)
    print(der[a]["input"] + der[a]["target"])
    n.train(der[a]["input"], der[a]["target"])

#print(n.feedForward([0,0]))
#print(n.feedForward([0,1]))
#print(n.feedForward([1,0]))
#print(n.feedForward([1,1]))

print("Input: " + str(n.inputWeights))
print("Hidden: " + str(n.hiddenWeights))
print("Ouput: " + str(n.outputWeights))