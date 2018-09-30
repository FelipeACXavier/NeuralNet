import numpy as np

class NeuralNet:
    # Initiaite the neural net
    def __init__(self, layers):
        self.inputs = layers[0] # int
        self.outputs = layers[1] # int
        self.length = len(layers)# int
        self.nodes = []
        self.error = []
        for _ in range(self.length - 1):
            self.nodes.append(1)
            self.error.append(1)
        self.learningRate = 0.5
        self.bias = [np.random.randn(y, 1) for y in layers[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]     
          
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
            # print("self.nodes: " + str(input))
            self.nodes[i] = self.sigmoid(np.dot(self.weights[i], input) + self.bias[i])
            input = self.nodes[i]
            
        return self.nodes[-1]

    # Used for the backpropagation
    def train(self, input, target):
        out = self.feedForward(input)
        # (output - target) --> finds the difference between what we want and what it calculated
        outputError = (np.asarray(target).reshape(len(target),1) - out)
        input = np.asarray(input).reshape(len(input),1)
        for i in range(self.length - 1):
            # Apply the derivative of sigmoid to 
            dydz = self.dsigmoid(self.nodes[self.length-2-i])
            if i is 0:
                # Hadamard product of the error in the guess and the dsigmoid of the output
                delta = np.multiply(dydz, outputError)
                # Matrix multiplication of the delta and input of that weight (the node before the last node)
                self.error[i] = self.learningRate * np.dot(delta, self.nodes[-2].T)         
                self.bias[self.length-2-i] += delta  
            elif i is self.length-2:
                # Hadamard product of the error in guess and the dsigmoid of the output
                outputError = np.dot(self.weights[self.length-1-i].T, outputError)
                delta = np.multiply(dydz, outputError) 
                self.error[i] = self.learningRate * np.dot(delta, input.T) 
                self.bias[self.length-2-i] += delta
            else:
                outputError = np.dot(self.weights[self.length-1-i].T, outputError)
                delta = np.multiply(dydz, outputError) 
                self.error[i] = self.learningRate * np.dot(delta, self.nodes[self.length-2-i].T) 
                self.bias[self.length-2-i] += delta
        self.update()

        return outputError

    # Add the necessary change in weights
    def update(self):
        for i in range(self.length - 1):
                self.weights[i] += self.error[self.length-2-i]
    
    # save weight and bias for further use
    def save(self, name):
        file = open(name, 'wb')
        # save to txt file (so we can read)
        name = name.split('.', 2)
        self.saveText(name[0]+'.txt')

        # save to npy file
        np.save(file, self.weights)
        np.save(file, self.bias)
        file.close()

    # save to readable txt file
    def saveText(self,name):
        file = open(name, 'w')
        file.write("Weights: " + str(self.weights))
        file.write('\n')
        file.write('Bias: ' + str(self.bias))
        file.close()
    
    # Load weights and bias to current network
    def load(self, name):
        file = open(name,'rb')
        self.weights = np.load(file)
        self.bias = np.load(file)
        file.close()

# ---------------------------------------------------------