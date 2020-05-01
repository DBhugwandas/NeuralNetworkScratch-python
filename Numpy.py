import numpy as np
import math
from numpy import genfromtxt
import math
import copy

# Sigmoid Function


def sigmoid(Input):

    Out = 1 / (1 + np.exp(-Input))
    return Out


my_data = genfromtxt('diabetes.csv', delimiter=',')[1:, :]
Y = my_data[:, -1]
X = my_data[:, :-1]


def initialize_network(size):

    # Intializing theta matrix
    Theta1 = np.random.random((size[1], size[0] + 1)) * 2 * 0.12 - 0.12
    Theta2 = np.random.random((size[2], size[1] + 1)) * 2 * 0.12 - 0.12

    # Initalizing Capital Delta, same Dimensions as Theta Matrix
    CDelta1 = np.random.random((size[1], size[0] + 1))
    CDelta2 = np.random.random((size[2], size[1] + 1))

    return Theta1, Theta2, CDelta1, CDelta2


def FeedForward(XVal, Theta1, Theta2, size):

    # -------- Feedfoward Network
    Input = np.insert(XVal, 0, 1).reshape(size[0] + 1, 1)

    # Layer 1

    Activation1 = sigmoid(np.dot(Theta1, Input))
    Activation1 = np.insert(Activation1, 0, 1).reshape(size[1] + 1, 1)

    # Output Layer
    Output = sigmoid(np.dot(Activation1.transpose(), Theta2.transpose()))

    return Output, Activation1, Input


def Train(X, Y, alpha, lamb):

    size = [8, 3, 1]

    # Initalize network
    Theta1, Theta2, CDelta1, CDelta2 = initialize_network(size)

    # Parameters
    m = X.shape[0]

    # Multiple Epochs
    for i in range(1000):

        Cost = 0

        # Iterating over the training examples
        for i in range(X.shape[0]):

            XVal = X[i, :]
            yVal = Y[i]

            # FeedFoward Network

            Output, Activation1, Input = FeedForward(XVal, Theta1, Theta2, size)

            # Cost Function
            Cost = Cost + np.sum((-yVal * np.log(Output) - (1 - yVal) * (1 - np.log(Output))))

            # Calculating the error terms
            OutError = Output - yVal

            HiddenError = np.dot(Theta2.transpose(), OutError) * (Activation1 * (1 - Activation1))
            HiddenError = np.delete(HiddenError, 0).reshape(size[1], 1)

            # Calculating the CDelta Accumulator - Excludes the error on the bias term
            CDelta1 = CDelta1 + np.dot(HiddenError, (Input.transpose()))
            CDelta2 = CDelta2 + np.dot(OutError, (Activation1.transpose()))

        # Calculating the partial derivative after iteration across training examples
        # We don't regularise the bias term therefore set value to 0

        Theta1_Adj = copy.deepcopy(Theta1)
        Theta1_Adj[:, 0] = 0
        Theta2_Adj = copy.deepcopy(Theta2)
        Theta2_Adj[:, 0] = 0

        PartDer1 = (1 / m) * CDelta1 + lamb / m * Theta1_Adj
        PartDer2 = (1 / m) * CDelta2 + lamb / m * Theta2_Adj

        # Performing gradient descent to adjust parameters
        Theta1 = Theta1 - alpha * PartDer1
        Theta2 = Theta2 - alpha * PartDer2

        # Total Cost at end of Epoch
        TotalCost = (1 / m) * Cost + ((lamb / 2 * m) * np.sum(Theta1) + np.sum(Theta2))

        print('Total Cost:', TotalCost)

    return Theta1, Theta2


T1, T2 = Train(X, Y, 0.001, 0.001)
