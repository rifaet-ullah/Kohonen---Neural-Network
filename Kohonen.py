import random
import pandas as pd
import numpy as np

class Kohonen:

    def __init__(self, n_node=2, eta=0.5):
        self.n_node = n_node
        self.eta = eta

    def train(self, trainData):
        trainData = np.array(trainData)
        self.nFeature = trainData.shape[1]
        self.nodes = [Node(self.nFeature, i) for i in range(self.n_node)]
        self.clusters = {}

        for data in trainData:
            distances = [node.get_distance(data) for node in self.nodes]
            for i in range(len(distances)-1):
                for j in range(i+1, len(distances)):
                    if self.nodes[i].distance > self.nodes[j].distance:
                        temp = self.nodes[i]
                        self.nodes[i] = self.nodes[j]
                        self.nodes[j] = temp
            minDistance = self.nodes[0].distance
            clusterName = self.nodes[0].nodeNumber
            weightToBeSaved = self.nodes[0].weight
            nNeighbour = 4
            while nNeighbour > 0 and minDistance > 0:
                nodesToBeUpdated = self.nodes[:nNeighbour+1]
                for node in nodesToBeUpdated:
                    node.weight += self.eta * (data - node.weight)
                distances = [node.get_distance(data) for node in self.nodes]
                for i in range(len(distances)-1):
                    for j in range(i+1, len(distances)):
                        if self.nodes[i].distance > self.nodes[j].distance:
                            temp = self.nodes[i]
                            self.nodes[i] = self.nodes[j]
                            self.nodes[j] = temp
                minDistance = self.nodes[0].distance
                weightToBeSaved = self.nodes[0].weight
                nNeighbour -= nNeighbour * self.eta
                nNeighbour = round(nNeighbour)
            keys = self.clusters.keys()
            if clusterName not in keys:
                self.clusters[clusterName] = [weightToBeSaved]
            else:
                self.clusters[clusterName].append(weightToBeSaved)

    def test(self, testData):
        testData = np.array(testData)
        predicted = [self.predict(data) for data in testData]
        return predicted

    def predict(self, data):
        clusterNames = list(self.clusters.keys())
        position = clusterNames[0]
        minDistance = self.distance(self.clusters[position][0], data)
        for clusterName in clusterNames:
            distance = min([self.distance(self.clusters[clusterName][i], data)] for i in range(len(self.clusters[clusterName])))
            if distance < minDistance:
                minDistance = distance
                position = clusterName
        return position

    def distance(self, a, b):
        return sum((a[i]-b[i])**2 for i in range(len(a)))

class Node:

    def __init__(self, nFeature, nodeNumber):
        self.weight = [round(random.random(), 2) for _ in range(nFeature)]
        self.nodeNumber = nodeNumber

    def get_distance(self, data):
        self.distance = sum((self.weight[i]-data[i])**2 for i in range(len(data)))
        return self.distance

data = pd.read_csv('Database.csv')
data = data.drop('Class', axis=1)
data = data.iloc[0:10, :]
model = Kohonen()
model.train(data.iloc[0:7])
predicted = model.test(data.iloc[7:10])
print(predicted)