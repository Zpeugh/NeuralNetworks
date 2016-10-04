import numpy as np
import math

class MLP:
    def __init__(self, shape, eta=0.15, momentum=.1, init_lower_bound=-1, init_upper_bound=1):

        self.shape = shape
        self.eta = eta
        self.a = momentum
        self.weights = []
        self.outputs = []
        self.deltas = []
        self.num_layers = len(shape) - 1

        # add input layer, plus 1 bias term
        self.outputs.append(np.ones(shape[0] + 1))

        # add the rest of the layers
        for i in range(1, len(shape)):
            self.outputs.append(np.ones(shape[i]))


        for i in range(0, len(self.outputs) - 1):
            layer = len(self.outputs[i])
            next_layer = len(self.outputs[i+1])
            weights = np.random.random(layer * next_layer)
            # weights = np.ones(layer * next_layer) * .5
            weight_range = init_upper_bound - init_lower_bound
            weights = weights * weight_range + init_lower_bound
            self.weights.append(weights.reshape((layer, next_layer)))
        # print("weights: {0}".format(self.weights))
        self.dw = [0,]*len(self.weights)


    # Given a neural network, activation function and inputs, the error vector is returned
    def __forward_pass(self, input_vector):

        x = input_vector[0]
        y = input_vector[1]

        # add 1 to end of inputs for bias term
        self.outputs[0][0:-1] = x

        # print("Layers: {0}".format(self.outputs))

        for i in range(1,len(self.shape)):
            # print("hidden layer: {0}".format(layer))
            summation = np.dot(self.outputs[i-1],self.weights[i-1])
            self.outputs[i] = np.array(self.act_func(summation))


            # print("new inputs: {0}".format(inputs))
        # print("\nExpected output: {0}".format(y))
        # print("Actual output: {0}".format(inputs))
        # print("Activation outputs: {0}".format(self.outputs))
        return self.outputs[-1]


    def __backpropogate(self, target):
        ''' Back propagate error related to target using lrate. '''

        deltas = []

        # Compute error on output layer
        error = target - self.outputs[-1]
        delta = error * self.backprop_func(self.outputs[-1])
        deltas.append(delta)
        # print("Delta K: {0}".format(deltas))
        # Compute error on hidden layers
        for i in range(len(self.shape)-2,0,-1):
            d_out = np.array(self.backprop_func(self.outputs[i]))
            # print("d_out: {0}:".format(d_out))
            # print("weights: {0}".format(self.weights[i].T))
            delta = d_out * np.dot(deltas[0],self.weights[i].T)
            # print("delta_j: {0}".format(delta))
            deltas.insert(0,delta)

        # Update weights
        for i in range(len(self.weights)):
            layer = np.atleast_2d(self.outputs[i])
            delta = np.atleast_2d(deltas[i])
            dw = np.dot(layer.T,delta)
            self.weights[i] += self.eta*dw + self.a*self.dw[i]
            self.dw[i] = dw
        # print("New weights: \n{0}".format(self.weights))



        # print("input Weights after: \n{0}".format(self.weights[-2]))
    def train(self, training_data, act_func, d_act_func, iterations=100000):

        self.act_func = act_func
        self.backprop_func = d_act_func

        for epoch in range(iterations):
            if (epoch % 1000 == 0):
                     print("Epoch: {0}/{1}".format(epoch, iterations))
            for training_sample in training_data:
                self.__forward_pass(training_sample)
                self.__backpropogate(training_sample[1])
                # print("Expected output: {0}".format(training_sample[1]))
                # print("Actual output: {0}".format(self.outputs[-1]))

    def test(self, training_data, act_func):
        print("Expected  |  Actual ")
        for training_sample in training_data:
            actual = self.__forward_pass(training_sample)
            print("  {0}  |  {1}  ".format(training_sample[1][0], actual[0]))

# Takes an array of inputs and applies the logistic sigmoid function to all
# of them, returning another array
# def sig(x):
#     return 1 / (1 + math.exp(-x))
#
# def sigmoid(x):
#     return map(d_sig, x)
#
# def d_sig(x):
#     return sigmoid(x)* (1 - sigmoid(x))
#
# def d_sigmoid(x):
#     return map(d_sig, x)
#

def sigmoid(x):
    return np.tanh(x)

def d_sigmoid(x):
    return 1.0-x**2
