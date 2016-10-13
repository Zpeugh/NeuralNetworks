import MLP
reload(MLP)
import math
import numpy as np

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

momentums = [0, 0.9]
etas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

number_of_epochs = np.zeros((len(momentums), len(etas)))
sum_errors = np.zeros((len(momentums), len(etas)))

for j,a in enumerate(momentums):
    for k, eta in enumerate(etas):
        nn = MLP.MLP((4,4,1), eta=eta, momentum=a)
        print("\n###########################")
        print("Learning rate: {0}".format(eta))
        print("Momentum: {0}".format(a))
        number_of_epochs[j][k] = nn.train(training_data, max_epoch=1000000)
        sum_errors[j][k] = nn.test(training_data)
