'''
    CSE 5526:    Lab 2
    Author:      Zach peugh
    Date:        10/11/2016
    Description: This is the test script for the RBF class
                 which generates 75 data points from a
                 function and then tests various numbers
                 of bases and etas, plotting results
'''
import RBFNet
reload(RBFNet)
from RBFNet import RBFNet
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist
from scipy.signal import gaussian

'''
    func            The function to sample from
    n               The number of samples
    x_interval      a tuple of (lower_bound, upper_bound)
                    for the input of the data
    noise_interval  a tuple of (lower_bound, upper_bound)
                    for the noise to add to the data
    returns         tuple of (X, Y) where X is the inputs
                    and Y is the list of corresponding  outputs
'''
def generate_noisy_data(func, n, x_interval, noise_interval):
    # get n inputs in x_interval
    X = np.random.random(n)
    x_range = x_interval[1] - x_interval[0]
    X = X * x_range + x_interval[0]

    # get n noise values in noise_interval
    noise = np.random.random(n)
    noise_range = noise_interval[1] - noise_interval[0]
    noise = noise * noise_range + noise_interval[0]

    Y = map(func, X + noise)
    return (X, Y)

'''
    h(x) = 0.5 + 0.4sin(2*pi*x)
'''
def sample_function(x):
    return 0.5 + 0.4*math.sin(2 * math.pi * x)


x, y = generate_noisy_data(sample_function, 75, (0,1), (-0.1,0.1))

bases = [2,4,7,11,16]
etas = [0.01,0.02]

test_x = np.linspace(0,1,200)
expected_y = map(sample_function, test_x)

for eta in etas:
    for num_bases in bases:
        rbf_net = RBFNet((1,num_bases,1))
        rbf_net.train(x, y, epochs=100)

        pred_y = rbf_net.test(test_x, expected_y)

        plt.clf()
        fig = plt.figure()
        plt.scatter(x,y)
        plt.scatter(test_x, pred_y, color='r')
        plt.title("Eta: {0}, Number of Bases: {1}".format(eta, num_bases))
        fig.savefig("results/base_{0}_ete_{1}.jpg".format(num_bases, eta))
#
#
#
# plt.scatter(clusters[:,0], clusters[:,1], color='red', marker="x")
# plt.show()
