import numpy as np
import math
from numpy import genfromtxt


my_data = genfromtxt('diabetes.csv', delimiter=',')[1:, :]
Y = my_data[:, -1]
X = my_data[:, :-1]

# Preparing the data to be iterated - Accepts in an array and returns a list of vectos


def DataPrep(Classes, yVal, XVal):

    y_elements = []
    X_elements = []

    for i in range(yVal.size):

        # Y Value Data Prep
        Ypos = int(yVal[i])
        YMatrix = np.zeros(Classes)

        if Classes == 1:
            YMatrix[0] = Ypos
            y_elements.append(YMatrix)
        else:
            YMatrix[Ypos - 1] = 1
            y_elements.append(YMatrix)

        # X Value Data Prep
        X_elements.append(XVal[i, :].transpose())

    return X_elements, y_elements


Xval, yval = DataPrep(2, Y, X)

print(Xval[1])
