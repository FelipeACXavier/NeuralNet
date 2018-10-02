from NeuralNet import  NeuralNet
import numpy as np

# Test cases for xor problem
der = [{"input":[0,0,0], "target":[0,0]},
        {"input":[0,1,1], "target":[1,0]},
        {"input":[1,0,0], "target":[1,1]},
        {"input":[1,1,1], "target":[0,0]}]

# Initialize network
n = NeuralNet([3, 3, 3, 2])
n.setLearningRate(0.8)
n.loadNpy('/home/felipe_xavier/Projects/NeuralNet/TrainData/threeInputsTest.npy')
runcycles = 10000

# for i in range(runcycles):
#      a = np.random.randint(0,4)
#      #print(der[a]["input"] + der[a]["target"])
#      n.train(der[a]["input"], der[a]["target"])

print(n.feedForward([0,0,0]))
print(n.feedForward([0,1,1]))
print(n.feedForward([1,0,0]))
print(n.feedForward([1,1,1]))

# n.saveTxt('threeInputsTest.txt')
# n.saveNpy('threeInputsTest.npy')