from NeuralNet import NeuralNet

n = NeuralNet([2, 2, 1])

n.load('xor.txt')
#print("weight: "+ str(n.weights))
#print("Bias: "+ str(n.bias))

""" print(n.feedForward([0,0]))
print(n.feedForward([0,1]))
print(n.feedForward([1,0]))
print(n.feedForward([1,1])) """