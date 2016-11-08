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

def randomly_partition()

def cross_validation(train_labels, train_data, test_labels, test_data, folds=5, c, a):
    accuracies = []
    for fold in range(folds):
        model = svm_train(train_labels, train_data, "-t 2 -c {0} -g {1}".format(c, a))
        p_labels, p_acc, p_vals = svm_predict(test_labels, test_data, model)
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
for i,c in enumerate(C_values):
    for j,a in enumerate(alphas):
        accuracies[i,j] = cross_validation(train_labels, train_data, test_labels, test_data, c, a)


print(accuracies)
