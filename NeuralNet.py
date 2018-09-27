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

        self.learningRate = 0.9

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
                self.hidden_calc[i] = np.asarray(self.sigmoid(np.add(self.hidden_calc[i], self.hiddenBias[0]))) 
            elif i is self.layers:
                self.hidden_calc[i] = np.dot( self.outputWeights[0], self.hidden_calc[i - 1])
                self.hidden_calc[i] = np.asarray(self.sigmoid(np.add(self.hidden_calc[i], self.outputBias))) 
            else:
                self.hidden_calc[i] = np.dot(self.hiddenWeights[i - 1], self.hidden_calc[i - 1])
                self.hidden_calc[i] = np.asarray(self.sigmoid(np.add(self.hidden_calc[i - 1], self.hiddenBias[i])))
        # print("Hidden_Calc: " + str(self.hidden_calc))
        return np.asarray(self.hidden_calc[self.layers])

    def train(self, inputs, target):
        # Output error
        self.out = self.feedForward(inputs)
        self.outputErrors = np.subtract(self.out, target)
        
        # Hidden layers node errors ------------------------------------------------------------------------
        for i in range(self.layers):
            if i is 0:
                self.hiddenErrors[i] = np.dot( self.outputWeights[0].T, self.outputErrors.T)
           # elif i is self.layers:
           #     self.hiddenErrors[i] = np.dot(self.inputWeights[0].T , self.hiddenErrors[i-1])
            else:
                self.hiddenErrors[i] = np.dot(self.hiddenErrors[i-1].T, self.hiddenWeights[self.layers-1-i].T)
        # --------------------------------------------------------------------------------------------------
        self.update()
        return self.hiddenErrors
    
    def update(self):
        # Update output weights
        a = self.learningRate * np.multiply(self.outputErrors, self.dsigmoid(self.hidden_calc[self.layers]))
        b = self.hidden_calc[self.layers - 1][np.newaxis]
        self.outputWeights -= np.dot(a.T, b)
        
        # Update hidden weights
        for i in range(self.layers - 1):
            a = self.learningRate * np.multiply(self.hiddenErrors[i], self.dsigmoid(self.hidden_calc[self.layers-1-i]))
            self.hiddenWeights[i] -= np.dot(a, self.hidden_calc[self.layers - 1])
        
        # Update input weights
        a = self.learningRate * np.multiply(self.hiddenErrors[self.layers-1], self.dsigmoid(self.hidden_calc[0]))
        #if len(a) == 1:
         #   a = a.item()
        self.inputWeights -= np.dot(a, self.imp)

    def Error(self, weight, which):
        for i in range(len(weight)):
            denominator += weight[i]
        return weight[which]/denominator

n = NeuralNet(2, 1, 2, 2)
print("Input: " + str(n.inputWeights))
print("Hidden: " + str(n.hiddenWeights))
print("Ouput: " + str(n.outputWeights))
#print(n.outputBias)
#print("Bias: " + str(n.hiddenBias))
#print("train data: " + str(n.feedForward([1,2])))
#for i in range(3):
#print(n.feedForward([1, 0]))
for i in range(100):
    n.train([1, 0], [1])
    print("Result: " + str(n.hidden_calc[n.layers]))
