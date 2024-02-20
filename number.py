import random
import math

class NeuralNetwork:

    neurons: int
    layers: int

    network: list
    weights: list
    bias: list

    def __init__(self, neurons: int, hiddenlayers: int):
        self.neurons = neurons
        self.layers = hiddenlayers

        self.network = self.create_network()
        self.weights = self.create_weights()
        self.bias = self.create_bias()

        return

    def create_network(self):
        network = []
        input = []
        network.append(input)

        for _ in range(0, self.layers):
            network.append(self.create_layer())

        output = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        network.append(output)

        return network

    def create_layer(self):
        layer = []
        for _ in range(0, self.neurons):
            layer.append(0)

        return layer

    def create_weights(self):
        weights = []
        for i in range(0, len(self.network) - 1):
            layerweights = []
            for _ in range(0, len(self.network[i + 1])):
                nodeweights = []
                for _ in range(len(self.network[i])):
                    nodeweights.append(2 * random.random() - 1)
                layerweights.append(nodeweights)
            weights.append(layerweights)
        return weights

    def create_bias(self):
        bias = []
        for i in range(0, len(self.network) - 1):
            nodebias = []
            for _ in range(0, len(self.network[i + 1])):
                nodebias.append(2 * random.random() - 1)
            bias.append(nodebias)
        return bias

    def calculate_layers(self):
        for i in range(1, len(self.network)):
            for j in range(0, len(self.network[i])):
                self.network[i][j] = 0
                for k in range(0, len(self.weights[i - 1][j])):
                    self.network[i][j] += self.weights[i - 1][j][k] * self.network[i-1][k]
                self.network[i][j] += self.bias[i - 1][j]
                self.network[i][j] = tanh(self.network[i][j])
        return

    def get_output(self):
        for i in range(0, len(self.network[-1])):
            if self.network[-1][i] == max(self.network[-1]):
                return i
        return -1

    def costFunction(self):
        return

def decimal_to_binary(number:int):
    binary = []
    for i in range(3, -1, -1):
        if number >= (2**i):
            binary.append(1)
            number -= (2**i)
        else:
            binary.append(0)
    return binary


def tanh(value: float):
    return (math.e**value - math.e**(-value))/(math.e**value + math.e**(-value))

def dtanh(value: float):
    return (2/(math.e**value + math.e**(-value)))**2


if __name__ == "__main__":

    neurons = 2
    layers = 2
    neuralnetwork = NeuralNetwork(neurons, layers)


