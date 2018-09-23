import numpy as np

# this NN has a baked-in 4-neuron middle layer

# input [x1, x2, x3, x4 ... xN]
# weights1 = {w_ij}, weight from xi to hidden neuron j
# weights2 = {u_ij}, weight from hidden neuron i to output j


class NeuralNetwork:
    def __init__(self, inputs, hiddenLayerNeurons, outputs):
        self.inputLayer     = np.zeros(inputs)
        self.hiddenLayer    = np.zeros(hiddenLayerNeurons)
        self.outputLayer    = np.zeros(outputs)
        self.weights1       = np.random.rand(hiddenLayerNeurons, inputs)
        self.weights2       = np.random.rand(outputs, hiddenLayerNeurons)
        
        
    def feedforward(self, inputArray):
        self.inputLayer = inputArray
        _hiddenLayer = np.dot(self.weights1, self.inputLayer)
        self.hiddenLayer = self.sigmoid(_hiddenLayer)
        print("\\\\\||||| INPUT -> HIDDEN LAYER |||||/////")
        print(self.weights1)
        print("DOT")
        print(self.inputLayer)
        print("EQUALS")
        print(_hiddenLayer)
        print("NORMALIZED")
        print(self.hiddenLayer)

        _outputLayer = np.dot(self.weights2, self.hiddenLayer)
        self.outputLayer = self.sigmoid(_outputLayer)

        print("\\\\\||||| HIDDEN LAYER -> OUTPUT LAYER |||||/////")
        print(self.weights2)
        print("DOT")
        print(self.hiddenLayer)
        print("EQUALS")
        print(_outputLayer)
        print("NORMALIZED")
        print(self.outputLayer)

    def sumsquares(self, vec1, vec2):
        if not vec1.size== vec2.size:
            print("ERROR [NeuralNetwork.sumsquares]: vectors not of same size:")
            print(vec1)
            print(vec2)
            return
        total = 0
        for i in range(vec1.size):
            total += np.power(vec1[i] - vec2[i], 2)
        return total
        
    ### mathematical functions
    def sigmoid(self, array):
        outArray = np.zeros(array.size)
        for i in range(array.size):
            outArray[i] = (1/(1+np.exp(-array[i])))
        return outArray
    def D_sigmoid(self, array):
        outArray = np.zeros(array.size)
        for i in range(array.size):
            outArray[i] = array[i]*(1.0-array[i])
        return outArray

NN = NeuralNetwork(3,3,3)
NN.feedforward([4,2,5])

