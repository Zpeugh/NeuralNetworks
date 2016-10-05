import MLP
import math
import numpy as np

reload(MLP)

# def sig(x):
#     return 1 / (1 + math.exp(-x))
#
# def sigmoid(x):
#     return map(sig, x)
#
# def d_sig(x):
#     return sig(x)* (1 - sig(x))
#
# def d_sigmoid(x):
#     return map(d_sig, x)

def sigmoid(x):
    ''' Sigmoid like function using tanh '''
    return np.tanh(x)

def d_sigmoid(x):
    ''' Derivative of sigmoid above '''
    return 1.0-x**2

training_data = [
                    [[0,0,0,0], [0]],
                    [[0,0,0,1], [1]],
                    [[0,0,1,0], [1]],
                    [[0,0,1,1], [0]],
                    [[0,1,0,0], [1]],
                    [[0,1,0,1], [0]],
                    [[0,1,1,0], [0]],
                    [[0,1,1,1], [1]],
                    [[1,0,0,0], [1]],
                    [[1,0,0,1], [0]],
                    [[1,0,1,0], [0]],
                    [[1,0,1,1], [1]],
                    [[1,1,0,0], [0]],
                    [[1,1,0,1], [1]],
                    [[1,1,1,0], [1]],
                    [[1,1,1,1], [0]]
                ]


nn = MLP.MLP((4,4,1), eta=0.15, momentum=0)
nn.train(training_data, sigmoid, d_sigmoid, iterations=40000)

nn.test(training_data, sigmoid)
