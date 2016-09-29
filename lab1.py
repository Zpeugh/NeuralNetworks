import numpy as np
import math


NUM_INPUTS = 4
HIDDEN_NEURONS = 4
OUTPUT_NEURONS = 1
INIT_LOWER_BOUND = -1
INIT_UPPER_BOUND = 1
ERROR_THRESHOLD = 0.05
ETA = 0.3
ALPHA = 1

training_data = [
                    [[0,0,0,0], 0],
                    [[0,0,0,1], 1],
                    [[0,0,1,0], 1],
                    [[0,0,1,1], 0],
                    [[0,1,0,0], 1],
                    [[0,1,0,1], 0],
                    [[0,1,1,0], 0],
                    [[0,1,1,1], 1],
                    [[1,0,0,0], 1],
                    [[1,0,0,1], 0],
                    [[1,0,1,0], 0],
                    [[1,0,1,1], 1],
                    [[1,1,0,0], 0],
                    [[1,1,0,1], 1],
                    [[1,1,1,0], 1],
                    [[1,1,1,1], 0]
                ]

def create_layer(num_inputs, hidden_neurons, init_lower_bound, init_upper_bound):
    total_inputs = num_inputs + 1 # add bias input
    weights = np.random.random(total_inputs * hidden_neurons)
    weight_range = init_upper_bound - init_lower_bound
    weights = weights * weight_range + init_lower_bound
    return weights.reshape((total_inputs, hidden_neurons))

def create_neural_network(layers):
    network = []
    for layer in layers:
        network.append(layer)
    return network

# Takes an array of inputs and applies the logistic sigmoid function to all
# of them, returning another array
def log_sig_act_func(x):
    return 1 / (1 + math.exp(-ALPHA * x))

def log_sig_backprop_func(x):
    return ALPHA * log_sig_act_func(x)* (1 - log_sig_act_func(x))

# Given a neural network, activation function and inputs, the error vector is returned
def forward_pass(network, act_func, input_vector):
    x = input_vector[0]
    y = input_vector[1]
    inputs = np.hstack((1, x)).reshape((1, len(x) + 1))
    # print("input: {0}".format(inputs))
    num_layers = len(network)

    for i, layer in enumerate(network):
        # print("hidden layer: {0}".format(layer))
        outputs = inputs.dot(layer).flatten()
        # print("dot product: {0}".format(outputs))
        inputs = map(act_func, outputs)
        if (i < num_layers - 1):
            inputs = np.hstack((1, inputs)).reshape((1, len(inputs) + 1))
        # print("new inputs: {0}".format(inputs))
    # print("\nExpected output: {0}".format(y))
    # print("Actual output: {0}".format(inputs))
    return y - np.array(inputs)

# Backpropogates all of the errors through the given neural network,
# updating weights and then returning the network
## TODO: Fix this shit.  Need to sum weights somewhere and do delta multiplication
##       and a lot more that I barely understand.
def update_weights(network, errors, backprop_func, eta):
    print("Network Layer 1: \n{0}".format(network[0]))
    print("Network Layer 2: \n{0}".format(network[1]))
    print("Error:\n{0}".format(errors))
    print("\n\nBEGIN BACK PROPOGRATION\n")
    new_weights = errors
    for i in range(1, len(network) + 1):
        layer = network[-i]
        new_layer = []
        print("layer: \n{0}".format(layer))
        error = 0.0
        for neuron_weight in layer.T:
            print("\tneuron weight: {0}".format(neuron_weight))
            new_weight = neuron_weight - np.array(map(backprop_func,neuron_weight)) * eta
            new_layer.append(new_weight)
        network[-i] = np.array(new_layer).T
        # print("\nNew Layer: \n{0}".format(np.array(new_layer).T))

    return network

def test_network(network, testing_data, act_func):
    for test_data in testing_data:
        x = test_data[0]
        y = test_data[1]
        inputs = np.hstack((1, x)).reshape((1, len(x) + 1))
        # print("input: {0}".format(inputs))
        num_layers = len(network)

        for i, layer in enumerate(network):
            outputs = inputs.dot(layer).flatten()
            inputs = map(act_func, outputs)
            if (i < num_layers - 1):
                inputs = np.hstack((1, inputs)).reshape((1, len(inputs) + 1))
            # print("new inputs: {0}".format(inputs))
        print("\nExpected output: {0}".format(y))
        print("Actual output: {0}".format(inputs))

################# Begin Script Execution ##################

# Initialize the neural network

hidden_layer = create_layer(NUM_INPUTS, HIDDEN_NEURONS, INIT_LOWER_BOUND, INIT_UPPER_BOUND)
output_layer = create_layer(HIDDEN_NEURONS, OUTPUT_NEURONS, INIT_LOWER_BOUND, INIT_UPPER_BOUND)

network = create_neural_network([hidden_layer, output_layer])
# print(network)

num_errors = 1
epoch = 0
x = training_data[0]
errors  = forward_pass(network, log_sig_act_func, x)
network = update_weights(network, errors, log_sig_backprop_func, ETA)
#
while(epoch < 10000):

    if (epoch % 1000 == 0):
        print("Epoch: {0}".format(epoch))
    errors = []
    for i, x in enumerate(training_data):
        error = forward_pass(network, log_sig_act_func, x)
        network = update_weights(network, error, log_sig_backprop_func, ETA)
        errors.append( abs(error[0]) < ERROR_THRESHOLD )
    epoch += i
    num_errors = sum(errors)
    # print(errors)
    if (epoch % 1000 == 0):
        print("Number of errors: {0}".format(num_errors))
    if (num_errors <= 0):
        print("0 errors..")
    if (epoch >= 100000):
        print("Beyond epoch")

print("\n\n\n\n")
test_network(network, training_data, log_sig_act_func)
