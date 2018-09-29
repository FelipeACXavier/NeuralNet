import numpy as np

class NeuralNet:
    # Initiaite the neural net
    def __init__(self, layers):
        self.inputs = layers[0] # int
        self.outputs = layers[1] # int
        self.length = len(layers)# int
        print(self.length)
        self.nodes = []
        self.error = []
        for _ in range(self.length - 1):
            self.nodes.append(1)
            self.error.append(1)
        self.learningRate = 0.5
        self.bias = [np.random.randn(y, 1) for y in layers[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]
        #print(self.bias)
        #print(self.weights)        
          
    # Starts with a rate of 0.5 but it can be changed using this function
    def setLearningRate(self, rate):
        self.learningRate = rate
    
    # Definition of sigmoid function to keep outputs between 0 and 1
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    # Derivative of sigmoid function (activation function)
    def dsigmoid(self, x):
        return x*(1-x)
    
    # Feed the input from layer to layer until the output
    def feedForward(self, input): 
        # Makes sure the imput is read as a column vector
        input = np.asarray(input).reshape(len(input), 1)
        
        # Calculates the result of each multiplication and stores it into a new vector
        for i in range(self.length - 1):
            self.nodes[i] = self.sigmoid(np.dot(self.weights[i], input) + self.bias[i])
            input = self.nodes[i]

    def train(self, input, target):
        # (output - target) --> finds the difference between what 
        outputError = (self.nodes[-1] - np.asarray(target).reshape(len(target),1))
        input = np.asarray(input).reshape(len(input),1)
        for i in range(self.length - 1):
            # Apply the derivative of sigmoid to 
            dydz = self.dsigmoid(self.nodes[self.length - 2 - i])
            # dzdW = np.dot(self.weights[-1], self.nodes[-2]) 
            if i is 0:
                # Hadamard product of the error in guess and the dsigmoid of the output
                delta3 = np.multiply(outputError, dydz)
                print("delta: " + str(delta3.shape))
                # Matrix multiplication of the delta and the value in eah node of that layer
                self.error[i] = np.dot(delta3, self.nodes[self.length - 3 - i].T)
            elif i is self.length - 2:
                print("delta: " + str(delta3.shape))
                print("dzdW: " + str(self.weights[0].shape))
                delta3 = np.dot(self.weights[0], delta3) * self.dsigmoid(dydz)
                self.error[i] = np.dot(delta3, input.T)
            else:
                return

        print("error: "+ str(self.error))
        print("weight: "+ str(self.weights))

        

# ---------------------------------------------------------
# ---------------------------------------------------------
# ---------------------------------------------------------
n = NeuralNet([2, 3, 2])
'''der = [{"input":[0,0], "target":[0]},
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

print("Input: " + str(n.inputWeights))
print("Hidden: " + str(n.hiddenWeights))
print("Ouput: " + str(n.outputWeights))'''
n.feedForward([1, 1])
n.train([1, 1], [0, 1])