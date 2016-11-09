'''
    CSE 5526:    Lab 3
    Author:      Zach peugh
    Date:        11/07/2016
    Description: This is a script to test various parameters associated
                 with SVM on a sample dataset.
'''



# import sys
# sys.path.append('C:\\Program Files\\LIBSVM\\libsvm-3.21\\python\\')
import sklearn.svm as SVM
import numpy as np
# from sklearn.svm import LinearSVC as Linear_SVM
import matplotlib.pyplot as plt
from svmutil import *


TRAINING_DATA = "../data/training_data.txt"
TESTING_DATA = "../data/test_data.txt"
PART_1_RESULTS = "../results/part_1.png"
TRAINING_PERCENT = 0.5

# Randomly shuffle the input data and then return a sample percent
# of it back as condensed arrays of labels and data.
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


def generate_partitions(labels, data, n_partitions):
    p_size = len(labels) / float(n_partitions)
    partitioned_labels = [ labels[int(round(p_size * x)): int(round(p_size * (x + 1)))] for x in range(n_partitions) ]
    partitioned_data = [ data[int(round(p_size * x)): int(round(p_size * (x + 1)))] for x in range(n_partitions) ]
    return partitioned_labels, partitioned_data

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


############################ Begin Script ############################

train_labels, train_data = svm_read_problem(TRAINING_DATA)
test_labels, test_data = svm_read_problem(TESTING_DATA)

C_values = [2**x for x in np.arange(-4,9,1)]
alphas = C_values
######################### Part 1: Linear SVMs #########################


# accuracies = []
# for c in C_values:
#     model = svm_train(training_labels, training_data, "-t 0 -c {0}".format(c))
#     p_labels, p_acc, p_vals = svm_predict(testing_labels, testing_data, model)
#     accuracies.append(p_acc[0])
#
# plt.scatter(C_values, accuracies)
# plt.xscale('log')
# plt.xlabel("log(C)")
# plt.ylabel("Prediction Accuracy")
# plt.title("Linear SVM Prediction Accuracy as a Function of C")
# plt.savefig(PART_1_RESULTS)
# plt.show()


########################## Part 2: RBF SVMs ###########################
accuracies = np.zeros((len(C_values), len(alphas)))
sampled_labels, sampled_data = sample_data(train_labels, train_data, TRAINING_PERCENT)

best_c = 0
best_a = 0
max_acc = 0

for i,c in enumerate(C_values):
    for j,a in enumerate(alphas):
        acc = cross_validation(sampled_labels, sampled_data, c, a)
        if (acc > max_acc):
            best_c = c
            best_a = a
            max_acc = acc
        accuracies[i,j] = acc


model = svm_train(train_labels, train_data, "-t 2 -c {0} -g {1}".format(best_c, best_a))
pred_labels, accuracy, pred_vals = svm_predict(test_labels, test_data, model)

print(accuracies)
print("Optimal C: {0}".format(best_c))
print("Optimal a: {0}".format(best_a))
print("Optimal Accuracy: {0}".format(accuracy))
