import numpy as np
import math

class MLP:
    def __init__(self, shape, eta=0.15, momentum=0, init_lower_bound=-1, init_upper_bound=1):

        self.shape = shape
        self.eta = eta
        self.a = momentum
        self.weights = []
        self.outputs = []
        self.prev_w_deltas = []

        # add input and hiden layers, plus 1 bias term for each
        for i in range(0, len(shape) - 1):
            self.outputs.append(np.ones(shape[i] + 1))

        # Add output layer
        self.outputs.append(np.ones(shape[-1]))

        for i in range(0, len(self.outputs) - 1):
            layer = len(self.outputs[i])
            next_layer = len(self.outputs[i+1])
            if (i < len(self.outputs) - 2):
                next_layer -= 1
            weights = np.random.random(layer * next_layer)
            # weights = np.ones(layer * next_layer) * .5
            weight_range = init_upper_bound - init_lower_bound
            weights = weights * weight_range + init_lower_bound
            self.weights.append(weights.reshape((layer, next_layer)))
        # print("weights: {0}".format(self.weights))
        self.weight_change = [0,]*len(self.weights)

        for i in range(0, len(self.weights)):
            self.prev_w_deltas.append(np.zeros_like(self.weights[i]))

    # Given a neural network, activation function and inputs, the error vector is returned
    def __forward_pass(self, input_vector):

        x = input_vector[0]
        y = input_vector[1]

        # add 1 to end of inputs for bias term
        self.outputs[0][1:] = x

        # print("Layers: \n{0}".format(self.outputs))
        # print("Weights: \n{0}".format(self.weights))

        for i in range(0,len(self.shape) - 1):
            # print("hidden layer: {0}".format(layer))

            # print("Inputs: {0}".format(self.outputs[i]))
            # print("Weights: {0}".format(self.weights[i]))

            # add a 1 for the bias node as an output for hidden layers
            output = self.act_func(np.dot(self.weights[i].T, self.outputs[i]))

            if (i < len(self.shape) - 2):
                output = np.hstack((1, output))

            self.outputs[i+1] = output

            # print("Outputs: \n{0}".format(output))

            # if (i == 0):
            #     outputs = np.atleast_2d(self.outputs[i][:-1]) # get all outputs except the bias term
            # else:
            #     outputs = np.array(self.outputs[i]).reshape((len(self.outputs[i]), 1))
            #
            # print("outputs:\n {0}".format(outputs.shape))
            # print(self.weights[i])
            # summation = np.dot(self.weights[i], outputs.T)
            # print("summation: \n{0}".format(summation))
            #
            # self.outputs[i] = np.stack(np.array(self.act_func(summation)), 1) # add the 1 for bias output


            # print("new inputs: {0}".format(inputs))
        # print("\nExpected output: {0}".format(y))
        # print("Actual output: {0}".format(inputs))
        # print("Activation outputs: {0}".format(self.outputs))
        return self.outputs[-1]


    def __backpropogate(self, target):

        deltas = []

        # Derive delta_k for output layer
        error = target - self.outputs[-1]
        delta_k = error * self.backprop_func(self.outputs[-1])
        deltas.append(delta_k)

        # Derive delta_j's for hidden layers
        for i in range(1, len(self.shape) - 1):
            output = self.outputs[-(i + 1)]
            d_out = np.array(self.backprop_func(output))
            # print("d_out: {0}:".format(d_out))
            # print("weights: {0}".format(self.weights[i].T))
            delta_j = d_out * np.dot(deltas[-i],self.weights[-i].T)
            # print("delta_j: {0}".format(delta))
            deltas.insert(0,delta_j[1:])

        # print("Deltas \n{0}".format(deltas))
        # print("Outputs: \n{0}".format(self.outputs))
        # Update weights
        # print("\n\n\n\n\n\n\n\n")
        # print("Deltas\n{0}".format(deltas))
        # print("Weights\n{0}".format(self.weights))
        # print("Outputs\n{0}".format(self.outputs))
        for j in range(0, len(self.weights)):
            # print("############### j = {0} ###############".format(j))
            for k in range(0, len(self.weights[j].T)):
                # print("j={0}, k={1}".format(j,k))
                # print("output: {0}".format(self.outputs[j]))
                # print("delta: {0}".format(deltas[j][k]))
                # print("weight: {0}".format(self.weights[j].T[k]))
                weight_change = self.eta * deltas[j][k] * self.outputs[j]
                # print("weight change: {0}".format(weight_change))
                self.weights[j].T[k] += weight_change
                # print("new weights: {0}".format(self.weights[j].T[k]))


    def train(self, training_data, act_func, d_act_func, iterations=1000000):

        self.act_func = act_func
        self.backprop_func = d_act_func

        epoch = 0
        errors = [True for i in range(len(training_data))]
        has_errors = True
        while (has_errors and epoch < iterations):
            epoch += 1
            if (epoch % 50 == 0):
                has_errors = sum(errors) != 0
            if (epoch % 10000 == 0):
                print("epoch {0}; {1} errors above 0.05".format(epoch, sum(errors)))
            if (epoch % 1000 == 0):
                     print("Epoch: {0}".format(epoch))
            for i,training_sample in enumerate(training_data):
                expected_output = training_sample[1][0]
                actual_output = self.__forward_pass(training_sample)
                self.__backpropogate(training_sample[1])
                errors[i] = abs(expected_output - actual_output[0]) > 0.05

            np.random.shuffle(training_data)


    def test(self, training_data, act_func):
        print("Expected  |  Actual ")
        for training_sample in training_data:
            actual = self.__forward_pass(training_sample)
            print("  {0}  |  {1}  ".format(training_sample[1][0], actual[0]))

# Takes an array of inputs and applies the logistic sigmoid function to all
# of them, returning another array
def sig(x):
    return 1 / (1 + math.exp(-x))

def sigmoid(x):
    return map(d_sig, x)

def d_sig(x):
    return sigmoid(x)* (1 - sigmoid(x))

def d_sigmoid(x):
    return map(d_sig, x)

#
# def sigmoid(x):
#     return np.tanh(x)
#
# def d_sigmoid(x):
#     return 1.0-x**2
