from NeuralNet import NeuralNet
import numpy as np

# Test cases for xor problem
der = [{"input":[0,0], "target":[0]},
        {"input":[0,1], "target":[1]},
        {"input":[1,0], "target":[1]},
        {"input":[1,1], "target":[0]}]
runcycles = 10000
# Initialize network
n = NeuralNet([2, 2, 1])

n.setLearningRate(0.8)

# Train the network
for i in range(runcycles):
    a = np.random.randint(0,4)
    #print(der[a]["input"] + der[a]["target"])
    n.train(der[a]["input"], der[a]["target"])

# Check if the results are satisfactory
print(n.feedForward([0,0]))
print(n.feedForward([0,1]))
print(n.feedForward([1,0]))
print(n.feedForward([1,1]))

n.saveNpy('xor.npy')
#n.saveText('xor.txt')