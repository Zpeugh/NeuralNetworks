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
from sklearn.svm import LinearSVC as Linear_SVM
import matplotlib.pyplot as plt


TRAINING_DATA = "../data/training_data.txt"
TESTING_DATA = "../data/test_data.txt"
PART_1_RESULTS = "../results/part_1.png"

# Read a line of n features and a label from a standard
# libsvm format input file
def read_line(line, expected_num_features=8):

    data = line.split(' ')
    features = np.zeros(expected_num_features)
    label = int(data[0])

    for i, feature in enumerate(data[1:]):
        tup = feature.split(':')
        features[int(tup[0])-1] = tup[1]

    return label, features

# Given the name of a file with data in standard libsvm compliant
# format, return an array of labels and an array of feature arrays
def import_data(file_name):
    data = []
    labels = []
    with open(file_name) as f:
        for line in f:
            label, features = read_line(line)
            labels.append(label)
            data.append(features)
    return np.array(labels, dtype=float), np.array(data, dtype=float)

def accuracy(predicted, actual):
    total = len(predicted)
    missed = np.count_nonzero(np.array(predicted) - np.array(actual))
    return 1 - (missed / float(total))

def cross_validation(train_labels, train_data, test_labels, test_data, folds=5):

    SVM.fit(train_data, train_labels, kernel='RBF', C=c)
    return 1

# Does
def cv_matrix(C_values, a_values):
    return 1


############################ Begin Script ############################



training_labels, training_data = import_data(TRAINING_DATA)

testing_labels, testing_data = import_data(TESTING_DATA)

C_values = [2**x for x in np.arange(-4,9,1)]
######################### Part 1: Linear SVMs #########################

# accuracies = []
# for c in C_values:
#     svm = Linear_SVM(C=c)
#     svm.fit(training_data, training_labels)
#     predicted_values = svm.predict(testing_data)
#     acc = accuracy(predicted_values, testing_labels)
#     print("C: {0} | Accuracy: {1}".format(c, acc))
#     accuracies.append(acc)
# plt.scatter(C_values, accuracies)
# plt.xscale('log')
# plt.xlabel("log(C)")
# plt.ylabel("Prediction Accuracy")
# plt.title("Linear SVM Prediction Accuracy as a Function of C")
# plt.savefig(PART_1_RESULTS)
# plt.show()


########################## Part 2: RBF SVMs ###########################
