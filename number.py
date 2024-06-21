import random
import numpy as np
from typing import List

class NeuralNetwork:

    def __init__(self, layers, neurons, learningrate) -> None:
        self.layers = layers
        self.neurons = neurons
        self.learningrate = learningrate

        self.weights = self.weight_matrix()
        self.bias = self.bias_matrix()
        self.postnetwork = self.network_matrix()
        self.prenetwork = self.network_matrix()

        self.weight_adjustments = np.array([])
        self.bias_adjustments = np.array([])
        return

    def weight_adjustment_matrix(self):
        matrix = []
        matrix.append(np.zeros((4, self.neurons)).T)
        for _ in range(0, self.layers - 1):
            matrix.append(np.zeros((self.neurons,self.neurons)).T)
        matrix.append(np.zeros((self.neurons,10)).T)

        return matrix

    def bias_adjustment_matrix(self):
        matrix = []
        for _ in range(0, self.layers):
            matrix.append(np.zeros(self.neurons))

        matrix.append(np.zeros(10))

        return matrix

    def weight_matrix(self) -> List:
        matrix = []
       
        matrix.append(np.random.randn(4, self.neurons).T)
        for _ in range(0, self.layers - 1):
            matrix.append(np.random.randn(self.neurons, self.neurons).T)

        matrix.append(np.random.randn(self.neurons, 10).T)

        return matrix

    def bias_matrix(self) -> List:
        matrix = []
        for _ in range(0, self.layers):
            matrix.append(np.random.randn(self.neurons))

        matrix.append(np.random.randn(10))

        return matrix

    def network_matrix(self):
        matrix = []

        for _ in range(0, self.layers):
            v = []
            for _ in range(0, self.neurons):
                v.append(0) 
            matrix.append(np.array(v))
        matrix.append(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))

        return matrix


    def feed_forward(self, input) -> None:
        self.prenetwork[0] = self.weights[0].dot(input) + self.bias[0]
        self.postnetwork[0] = self.relu(self.prenetwork[0])
        for i in range(1, self.layers):
            self.prenetwork[i] = self.weights[i].dot(self.postnetwork[i-1]) + self.bias[i]
            self.postnetwork[i] = self.relu(self.prenetwork[i])

        
        self.prenetwork[self.layers] = self.weights[self.layers].dot(self.postnetwork[self.layers - 1]) + self.bias[self.layers]
        self.postnetwork[self.layers] = self.softmax(self.prenetwork[self.layers])
        return
    
    def backpropagation(self, input) -> None:

        for i in range(0, len(batch)):
            
        

        self.weight_adjustments = 
        self.bias_adjustments = 

        return 

    def adjust(self) -> None:
        for i in range(0, len(self.weights)):
            sum = 0
            for j in range(0, len(self.weight_adjustments)):
                sum += self.weight_adjustments[i][j] 
            self.weights[i] -= self.learningrate * (1/self.weight_adjustments) * sum 
        for i in range(0, len(self.bias)):
            sum = 0
            for j in range(0, len(self.bias_adjustments)):
                sum += self.weight_adjustments[i][j] 
            self.bias[i] -= self.learningrate * (1/self.bias_adjustments) * sum
            return
    
    def relu(self, z) -> None:
        return np.maximum(0, z)

    def deriv_relu(self, z) -> None:
        return z > 0

    def softmax(self, z) -> None:
        return np.exp(z)/np.sum(np.exp(z))

    def generate_batches(self, num) -> List:
        batches = []
        for _ in range(0, num):
            batch = []
            for _ in range(0, 100):
                rand = random.randint(0, 9)
                batch.append(rand)
                batch.append(self.dcb(rand))
            batches.append(batch)

        return batches

    def dcb(self, input):
        binary = []
        for _ in range(0, 4):
            quotient = input//2
            remainder = input % 2
            binary.insert(0, remainder)
            input = quotient

        return np.array(binary)

    def bcd(self, binary):
        value = np.array([8, 4, 2, 1]).T
        return np.sum(binary.dot(value))

    def train(self, batch_num):
        batches = n.generate_batches(batch_num)
        for i in range(0, len(batches)):
            for j in range(0, len(batches[i])):
                n.feed_forward(batches[i][j])
                n.backpropagation(batches[i][j])
            n.adjust()
        return

if __name__ == "__main__":
    n = NeuralNetwork(2, 4, 1)
    n.train(10)
