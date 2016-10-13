'''
    CSE 5526:    Lab 2
    Author:      Zach peugh
    Date:        10/11/2016
    Description: This a Radial Basis Function Network class
                 which can be trained to estimate a function
'''
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist
from scipy.signal import gaussian

class RBFNet:

    def __init__(self, shape, eta=0.01):
        '''
            A Radial Basis Function Neural Network

            shape   tuple of integers as (input, centers, outputs)
            eta     the learning parameter
        '''
        self.n_inputs = shape[0]
        self.n_centers = shape[1]
        self.n_outputs = shape[2]
        self.centers = []
        self.gaussian_widths = np.zeros((1,self.n_centers))
        self.activations = np.ones((1,self.n_centers + 1))
        self.weights = np.random.random((self.n_centers + 1, self.n_outputs)) * 0.2 - 0.2
        self.eta = eta


    def _gaussian_activations(self, x):
        '''
            sigma   The scalar gaussian width of the cluster
            x       scalar x input where n is number of inputs
            mu      scalar center x coordinate of cluster

            returns (1, n_centers) gaussian activations
        '''
        beta = -1/(2*self.gaussian_widths**2)
        # print("beta", beta)
        distances = (self.centers - x)**2
        # print("sum_distance", distances)
        return np.exp(beta*distances)


    def _build_bases(self,x,y):
        km_clusterer = KMeans(n_clusters=self.n_centers)
        km_clusterer.fit(np.array(zip(x,y)))
        # self.centers = km_clusterer.cluster_centers_
        labels = km_clusterer.labels_

        unique_labels = np.unique(labels)
        cluster_groups = [[] for label in np.unique(labels)]
        for i, label in enumerate(labels):
            cluster_groups[label].append((x[i], y[i]))
        for i, cluster in enumerate(cluster_groups):
            self.centers.append(np.mean(cluster,axis=0)[0])
            self.gaussian_widths[0,i] = np.std(cluster)


    def _forward_pass(self,x, y):

        self.activations[0,1:] = self._gaussian_activations(x).flatten()
        # print(self.activations)
        return self.activations.dot(self.weights)

    def train(self, x, y, epochs=1):
        '''
            X   matrix with shape (n, input-dimension)
            Y   matrix with shape (n, output-dimension)
        '''
        self._build_bases(x,y)

        for epoch in range(epochs):
            for i in range(len(x)):
                d = self._forward_pass(x[i], y[i])

                deltas = self.eta * (d - y[i]) * self.activations
                self.weights = self.weights - deltas
                print(self.weights)


        # for i,center in enumerate(self.centers):
        #     x_j = center[0]
        #     y_j = center[1]
        #     self.activations[0,i] = self._gaussian_activation(self.gaussian_widths[i], x, x_j)
        #
        #
        # for i,x in enumerate(x):


        return 1


    def test(self, data):
        return 1
