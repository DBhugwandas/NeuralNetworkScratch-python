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

    Theta = []
    CDelta = []

    # Intializing theta matrix and Gradient Accumulator
    for i in range(len(size) - 1):
        Theta.append(np.random.random((size[i + 1], size[i] + 1)) * 2 * 0.12 - 0.12)
        CDelta.append(np.zeros((size[i + 1], size[i] + 1)))

    return Theta, CDelta[0], CDelta[1]


def FeedForward(XVal, Theta, size):

    # -------- Feedfoward Network
    Input = np.insert(XVal, 0, 1).reshape(size[0] + 1, 1)

    Activations = []
    # Layer 1
    for i in range(len(size) - 1):

        # First Layer Hidden
        if i == 0:
            Activation1 = sigmoid(np.dot(Theta[0], Input))
            Activation1 = np.insert(Activation1, 0, 1).reshape(size[1] + 1, 1)
            Activations.append(Activation1)

        # Other Hidden Layers
        if i < len(size) - 2 and i > 0:
            Activation2 = sigmoid(np.dot(Theta[i - 1], Activations[i - 1]))
            Activation2 = np.insert(Activation1, 0, 1).reshape(size[i + 1] + 1, 1)
            Activations.append(Activation2)

        # Output Layers
        else:
            Output = sigmoid(np.dot(Activation1.transpose(), Theta[-1].transpose()))

    return Output, Activations, Input


def CostFunction(yVal, Output):
    Cost = np.sum((-yVal * np.log(Output) - (1 - yVal) * (1 - np.log(Output))))
    return Cost


def Train(X, Y, alpha, lamb, size):

    # Initalize network
    Theta, CDelta1, CDelta2 = initialize_network(size)

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
            Output, Activations, Input = FeedForward(XVal, Theta, size)

            # Cost Function
            Cost += CostFunction(yVal, Output)

            # Calculating the error terms

            # Output Error
            OutError = Output - yVal

            # Hidden Layer Error
            HiddenError = np.dot(Theta[1].transpose(), OutError) * (Activations[0] * (1 - Activations[0]))
            HiddenError = np.delete(HiddenError, 0).reshape(size[1], 1)

            # Calculating the CDelta Accumulator - Excludes the error on the bias term
            CDelta1 = CDelta1 + np.dot(HiddenError, (Input.transpose()))
            CDelta2 = CDelta2 + np.dot(OutError, (Activations[0].transpose()))

        # Calculating the partial derivative after iteration across training examples
        # We don't regularise the bias term therefore set value to 0

        # Performing gradient Descent - CHECK IF CODE IF WORKING
        CDelta = [CDelta1, CDelta2]

        for i in range(len(Theta)):
            Theta_temp = copy.deepcopy(Theta[i])
            Theta_temp[:, 0] = 0

            PartDer = (1 / m) * CDelta[i] + lamb / m * Theta_temp
            Theta[i] = Theta[i] - alpha * PartDer

        # Total Cost at end of Epoch
        TotalCost = (1 / m) * Cost + ((lamb / 2 * m) * np.sum(Theta[0]) + np.sum(Theta[1]))

        print('Total Cost:', TotalCost)

    return Theta


T1, T2 = Train(X, Y, 0.001, 0.001, [8, 3, 1])
