'''
    CSE 5526:    Lab 3
    Author:      Zach peugh
    Date:        11/07/2016
    Description: This is a script to test various parameters associated
                 with SVM on a sample dataset.  Part 1 entails A Linear
                 SVM while part 2 deals with a Radial Basis Function kernel
'''

import numpy as np
import matplotlib.pyplot as plt
from svmutil import *

TRAINING_DATA = "../data/training_data.txt"
TESTING_DATA = "../data/test_data.txt"
PART_1_RESULTS = "../results/part_1.png"
PART_2_RESULTS = "../results/part_2.png"
TRAINING_PERCENT = 0.5

# Return a random subset of label and data arrays
def sample_data(labels, data, percent):
    tuples = zip(labels, data)
    shuffled_labels = []
    shuffled_data = []
    np.random.shuffle(tuples)
    for tup in tuples:
        shuffled_labels.append(tup[0])
        shuffled_data.append(tup[1])

    n_samples = int(len(labels) * percent)
    return shuffled_labels[:n_samples], shuffled_data[:n_samples]

# Returns an array of n partitions for labels
# and an array of n partitions for the data
def generate_partitions(labels, data, n):
    p_size = len(labels) / float(n)
    partitioned_labels = [ labels[int(round(p_size * x)): int(round(p_size * (x + 1)))] for x in range(n) ]
    partitioned_data = [ data[int(round(p_size * x)): int(round(p_size * (x + 1)))] for x in range(n) ]
    return partitioned_labels, partitioned_data

# Performs N-fold Cross validation, returning the average
# classification acuracy on the training data
def cross_validation(train_labels, train_data, c, a, folds=5):
    accuracies = []
    p_labels, p_data = generate_partitions(train_labels, train_data, folds)
    for i in range(folds):
        train_labels = []
        train_data = []
        for j in range(len(p_labels)):
            if j is not i:
                train_labels = np.concatenate((train_labels,p_labels[j]))
                train_data = np.concatenate((train_data,p_data[j]))
        model = svm_train(list(train_labels), list(train_data), "-t 2 -c {0} -g {1}".format(c, a))
        pred_labels, p_acc, pred_vals = svm_predict(p_labels[i], p_data[i], model)
        accuracies.append(p_acc[0])
    return np.mean(accuracies)

# Prints out and saves the graph of accuracy as a function of c_values
def print_linear_svm_accuracies(accuracies, C_values, file_name):
    plt.clf()
    plt.figure(figsize=(10,6))
    plt.scatter(C_values, accuracies)
    plt.xscale('log')
    plt.xlabel("log(C)")
    plt.ylabel("Prediction Accuracy")
    plt.title("Linear SVM Prediction Accuracy as a Function of C")
    plt.savefig(file_name)
    plt.show()
    plt.clf()

# Prints and saves a confusion matrix style colored plot for the given matrix
def print_accuracy_matrix(accuracies, alphas, c_values, file_name):
    plt.clf()
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(accuracies, cmap=plt.cm.jet, interpolation='nearest')
    fig.colorbar(res)
    plt.xticks(range(len(alphas)), alphas)
    plt.yticks(range(len(c_values)), c_values)
    plt.title("Cross Validation Accuracy For Varying C and a Values")
    plt.ylabel("C Values")
    plt.xlabel("alpha Values")
    plt.savefig(file_name)
    fig.show()

############################ Begin Script ############################

train_labels, train_data = svm_read_problem(TRAINING_DATA)
test_labels, test_data = svm_read_problem(TESTING_DATA)

C_values = [2**x for x in np.arange(-4,9,1)]
alphas = C_values


######################### Part 1: Linear SVMs #########################
p1_accuracies = []
for c in C_values:
    model = svm_train(train_labels, train_data, "-t 0 -c {0}".format(c))
    p_labels, p_acc, p_vals = svm_predict(test_labels, test_data, model)
    p1_accuracies.append(p_acc[0])

print("\n\n\n#################### Linear SVM Results ####################")
print("Optimal C: {0}".format(C_values[np.argmax(p1_accuracies)]))
print("Achieved Accuracy: {0}".format(np.max(p1_accuracies)))

print_linear_svm_accuracies(p1_accuracies, C_values, PART_1_RESULTS)


########################## Part 2: RBF SVMs ###########################
p2_accuracies = np.zeros((len(C_values), len(alphas)))
sampled_labels, sampled_data = sample_data(train_labels, train_data, TRAINING_PERCENT)
best_c, best_a, max_acc = 0, 0, 0

for i,c in enumerate(C_values):
    for j,a in enumerate(alphas):
        acc = cross_validation(sampled_labels, sampled_data, c, a)
        if (acc > max_acc):
            best_c = c
            best_a = a
            max_acc = acc
        p2_accuracies[i,j] = acc

model = svm_train(train_labels, train_data, "-t 2 -c {0} -g {1}".format(best_c, best_a))
pred_labels, p2_accuracy, pred_vals = svm_predict(test_labels, test_data, model)

print("\n\n\n#################### RBF SVM Results ####################")
print("Optimal C: {0}".format(best_c))
print("Optimal a: {0}".format(best_a))
print("Achieved Accuracy: {0}".format(p2_accuracy[0]))

print_accuracy_matrix(p2_accuracies, alphas, C_values, PART_2_RESULTS)
