#!usr/bin/env python
# Felipe A. C. Xavier
# Neural Network first implementation

import numpy as np

class NeuralNet:
    # Initialize Neural Network
    # Number of inputs and outputs need to be specified as lists
    def __init__(self, inputs, outputs, layers, nodes):
        self.input = inputs
        print(self.input)
        self.output = outputs
        print(self.output)
        self.outbias = 0
        self.nodes = nodes
        self.learningRate = 0.5
        self.setHiddenLayers(layers, nodes)
        self.setBias(nodes)
        self.setWeights(self.neurons)

    # Starts with a rate of 0.5 but it can be changed using this function
    def setLearnRate(self, rate):
        self.learningRate = rate
    # ------------------------------------------------------------- #
    # -----------  Initialize values for whole network   ---------- #
    # ------------------------------------------------------------- #

    # Add hidden layers to the Neural Network and initialize them with random values
    # layers is the number of layers
    # nodes is the number of neurons per layer
    def setHiddenLayers(self, layers, nodes):
        self.layers = layers
        # Initialize an empty list of neurons
        self.neurons = []
        # Pass through every layer
        for i in range(self.layers):
            self.neurons.append([])
            # Pass through every node in each layer
            for j in range(nodes):
                self.neurons[i].append(np.random.rand())
        print("Values set")

    # Set the biases for every neuron
    def setBias(self, nodes):
        # Initialize an empty list of neurons
        self.biases = []
        # Pass through every layer
        for i in range(self.layers):
            self.biases.append([])
            # Pass through every node in each layer
            for j in range(nodes):
                self.biases[i].append(np.random.rand())
            self.outbias = np.random.rand()
        print("Biases set")
    
    # Calculate the cost of each train session
    def cost(self):
        costs = 0.0
        length = len(self.output)
        for i in range(len(self.output)):
            costs = costs + (self.output[i] - self.target)**2
        return costs/length
    # ------------------------------------------------------------- #
    # ----------  Initialize weights for whole network   ---------- #
    # ------------------------------------------------------------- #

    # Create and initialize weights
    # Each set is a vector of vectors which represent the weights of
    # all nodes of the previous layer in relation to that neuron
    def setWeights(self, neurons):
        # ---------------------------------------------------------
        # weightInput is a (inputs x first layer) matrix
        self.weightInput = []
        # Go through every node of input
        for i in range(len(self.input)):
            self.weightInput.append([])
            # Cycle through every neuron for each input
            for j in range(len(self.neurons[0])):
                self.weightInput[i].append(np.random.rand())

        # ---------------------------------------------------------
        # weightHidden is a layers*(neurons x neurons) matrix
        self.weightHidden = []
        # Cycle through each layer
        for i in range(self.layers - 1):
            self.weightHidden.append([])
            # Cycle through each neuron of each layer
            for j in range(len(self.neurons[i])):
                self.weightHidden[i].append([])
                # Cycle through each neuron of each other neuron for each layer
                for k in range(len(self.neurons[i])):
                    self.weightHidden[i][j].append(np.random.rand())

        # ---------------------------------------------------------
        # weightOutput is a (last layer x output) matrix
        self.weightOutput = []
        # Cycle through each neuron of last layer
        for i in range(len(self.output)):
            self.weightOutput.append([])
            # Cycle through every output for each last layer neuron
            for j in range(self.nodes):
                self.weightOutput[i].append(np.random.rand())
        print("Weights set")
   
    # ------------------------------------------------------------- #
    # --------------    To be used within the class    ------------ #
    # ------------------------------------------------------------- #

    # Definition of sigmoid function to keep outputs between 0 and 1
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    # Matrix multiplication to be used in guess()
    def multiply(self, a, b):
        # Check if it is possible to perform the multiplication
        if np.shape(a)[0] != np.shape(b)[1]:
            print("Cannot perform operation! Rows of A need to be equal to columns of B")
        else:
            result = []
            # Go through rows of a
            for i in range(np.shape(a)[0]):
                # Go through columns of b
                for j in range(np.shape(b)[1]):
                    sum = 0.0 
                    # Go through columns of a to add everything and then save to the new array
                    for k in range(np.shape(a)[1]):
                        sum = sum + a[i][k]*b[k][j]
                    result.append(sum)
            return result
                
    # Calculate the error of the guess
    def error(self, output):
        dEdout = -(self.target - output)
        doutdnet = output*(1 - output)
        # dnetdw = output
        return dEdout + doutdnet + output
    
    # Update values based on previous generation
    def nextGeneration(self):
        return
    # ------------------------------------------------------------- #
    # --  Redefine biases and weight to better fit the results   -- #
    # ------------------------------------------------------------- #

    # Update the Neural Network based on the error calculated
    def learn(self):
        # ---------------------------------------------------------
        # Cycle through each neuron of last layer
        for i in range(len(self.output)):
            # Cycle through every output for each last layer neuron
            for j in range(self.nodes):
                self.weightOutput[i][j] = self.weightOutput[i][j] - self.learningRate * self.error(self.output[i])

        # ---------------------------------------------------------
        # Cycle through each layer
        for i in range(self.layers - 1):
            # Cycle through each neuron of each layer
            for j in range(len(self.neurons[i])):
                # Cycle through each neuron of each other neuron for each layer
                for k in range(len(self.neurons[i])):
                    self.weightHidden[i][j][k] = self.weightHidden[i][j][k] - self.learningRate * self.error(self.neurons[i + 1][k])      

        # ---------------------------------------------------------        
        # Go through every node of input
        for i in range(len(self.input)):
            # Cycle through every neuron for each input
            for j in range(len(self.neurons[0])):
                self.weightInput[i][j] = self.weightInput[i][j] - self.learningRate * self.error(self.neurons[0][j])

    # Guess the value
    def guess(self, expected):
        self.target = expected
        # ---------------------------------------------------------
        for i in range(len(self.input)):
            for j in range(len(self.neurons[0])):
                self.neurons[0][j] = self.neurons[0][j] + (self.weightInput[i][j] * self.input[i])
            self.neurons[0][j] = self.sigmoid(self.neurons[0][j] + self.biases[0][j]) 
        
        # ---------------------------------------------------------
        # Cycle through each layer
        for i in range(self.layers - 1):
            # Cycle through each neuron of each layer
            for j in range(len(self.neurons[i + 1])):
                # Cycle through each neuron of each other neuron for each layer
                for k in range(len(self.neurons[i])):
                    self.neurons[i + 1][j] = self.neurons[i + 1][j] + (self.weightHidden[i][j][k] * self.neurons[i][j])
                self.neurons[i + 1][j] = self.sigmoid(self.neurons[i + 1][j] + self.biases[i][j])

        # ---------------------------------------------------------
        # Cycle through each neuron of last layer
        for i in range(len(self.output)):
            # Cycle through every output for each last layer neuron
            for j in range(len(self.neurons[self.layers - 1])):
                self.output[i] = self.output[i] + (self.weightOutput[i][j] * self.neurons[self.layers - 1][j])
            self.output[i] = self.sigmoid(self.output[i] + self.outbias)
        
        return self.output

# Checking outputs
layer = 2
expect = 1
a = np.array([[1, 2, 3],[5, 8, 6], [1, 7, 5]])
b = np.array([[3, 4, 2],[4, 8, 5], [3, 7, 1]])
n = NeuralNet([1, 2],[1, 2], layer, 3)
for i in range(10):
    print("OUTPUT: " + str(n.guess(expect)))
    n.learn()
    print("Cost: " + str(n.cost()) + "\n")

    

