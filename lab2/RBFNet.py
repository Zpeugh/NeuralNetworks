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

    def __init__(self, shape, eta=0.01, standardize_variance=False):
        '''
            A Radial Basis Function Neural Network

            shape                   tuple of integers as
                                    (input, centers, outputs)
            eta                     the learning parameter
            standardize_variance    boolean to use an even gaussian
                                    width for all clusters or not
        '''
        self.n_inputs = shape[0]
        self.n_centers = shape[1]
        self.n_outputs = shape[2]
        self.centers = []
        self.gaussian_widths = np.ones((1,self.n_centers))
        self.activations = np.ones((1,self.n_centers + 1))
        self.weights = np.random.random((self.n_centers + 1, self.n_outputs)) * 0.2 - 0.2
        self.eta = eta
        self.standardize_variance = standardize_variance


    def _gaussian_activations(self, x):
        '''
            x       scalar x input where n is number of inputs

            returns (1, n_centers) gaussian activations
        '''
        beta = -1/(2*self.gaussian_widths**2)
        distances = (self.centers - x)**2
        return np.exp(beta*distances)


    def _build_bases(self,x,y):
        km_clusterer = KMeans(n_clusters=self.n_centers)
        km_clusterer.fit(np.array(zip(x,y)))
        labels = km_clusterer.labels_
        centers = km_clusterer.cluster_centers_

        if self.standardize_variance == True:
            d_max = np.max(pdist(centers, metric="euclidean"))
            gaus_width = d_max / np.sqrt(2 * self.n_centers)
            self.gaussian_widths = self.gaussian_widths * gaus_width

        unique_labels = np.unique(labels)
        cluster_groups = [[] for label in np.unique(labels)]
        for i, label in enumerate(labels):
            cluster_groups[label].append((x[i], y[i]))
        for i, cluster in enumerate(cluster_groups):
            self.centers.append(np.mean(cluster,axis=0)[0])
            if self.standardize_variance == False:
                self.gaussian_widths[0,i] = np.std(cluster)



    def _forward_pass(self,x, y):

        self.activations[0,1:] = self._gaussian_activations(x).flatten()
        # print(self.activations)
        return self.activations.dot(self.weights)

    def train(self, x, y, epochs=100):
        '''
            X       matrix with shape (n, input-dimension)
            Y       matrix with shape (n, output-dimension)
            epochs  number of times to run through the whole training set
        '''
        self._build_bases(x,y)

        for epoch in range(epochs):
            for i in range(len(x)):
                d = self._forward_pass(x[i], y[i])
                error = (d - y[i])
                deltas = self.eta * error * self.activations
                self.weights = self.weights - deltas.T


    def test(self, x, y):
        sum_error = 0
        predicted_values = []
        for i in range(len(x)):
            d = self._forward_pass(x[i], y[i])
            predicted_values.append(d)
            sum_error += abs(d - y[i])
        print("Average error: ", sum_error / len(x))
        return predicted_values
