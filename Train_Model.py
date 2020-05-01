# Creating a Neural Network training algorithm
import numpy as np
import copy
import math


class NeuralNetwork():

    # Defining model architecture
    def __init__(self, NbrNeurons):

        self.NbrNeurons = NbrNeurons

        # Creating a list of theta matrices and intializing them
        self.ThetaMatrix = [(np.random.random((self.NbrNeurons[i + 1], self.NbrNeurons[i] + 1)) * 2 * 0.12 - 0.12) for i in range(len(self.NbrNeurons) - 1)]

        # Creating the Activation Node Matrix including the bias term
        self.ActivationMatrix = [np.zeros((self.NbrNeurons[i] + 1, 1)) if i <= (len(self.NbrNeurons) - 2) else np.zeros((self.NbrNeurons[i], 1)) for i in range(len(self.NbrNeurons))]

        for i in range(len(self.ActivationMatrix) - 1):
            self.ActivationMatrix[i][0] = 1

        # Error Matrix Exluding bias terms
        self.ErrorMatrix = [np.zeros((self.NbrNeurons[i + 1], 1)) for i in range(len(self.NbrNeurons) - 1)]

        # Gradient Accumulator - same dimensions as theta matrix
        self.GradientAcc = [np.zeros((self.NbrNeurons[i + 1], self.NbrNeurons[i] + 1)) for i in range(len(self.NbrNeurons) - 1)]

    def DataPrep1(self, XVal, yVal):

        y_elements = []
        X_elements = []

        for i in range(yVal.size):

            # Y Value Data Prep
            Ypos = int(yVal[i])
            YMatrix = np.zeros(self.NbrNeurons[-1])

            if self.NbrNeurons[-1] == 1:
                YMatrix[0] = Ypos
                y_elements.append(YMatrix)
            else:
                YMatrix[Ypos - 1] = 1
                y_elements.append(YMatrix)

            # X Value Data Prep
            X_elements.append(XVal[i, :].transpose())

        return X_elements, y_elements

    # Training the model and performing backpropagation
    def Train(self, XVal, yVal, lamb, alpha, iters):

        # Multiple training iteractions on the training set
        self.m = XVal.shape[0]

        X_elements, y_elements = self.DataPrep1(XVal, yVal)

        for obs in range(len(X_elements)):

            XVal = X_elements[obs].reshape(self.NbrNeurons[0], 1)
            print(XVal.size)
            yVal = y_elements[obs].reshape(self.NbrNeurons[-1], 1)

            # ----- Item iteration loop should start here

            # Setting the X_Val as the first Activation Layer - Note the first term is a bias
            self.ActivationMatrix[0][1:] = XVal

            # FeedFoward Network - One observation at a time
            for i in range(len(self.NbrNeurons) - 1):

                # For the last matrix/output matrix there is no bias term
                # Hence we don't need to only the second element onwards (no bias in output)

                if i < len(self.NbrNeurons) - 2:

                    self.ActivationMatrix[i + 1][1:] = np.dot(self.ThetaMatrix[i], self.ActivationMatrix[i])
                    # Applying the sigmoid function
                    self.ActivationMatrix[i + 1][1:] = 1 / (1 + np.exp(-self.ActivationMatrix[i + 1][1:]))

                # If output node we dont have a bias term
                elif i == len(self.NbrNeurons) - 2:
                    self.ActivationMatrix[i + 1] = np.dot(self.ThetaMatrix[i], self.ActivationMatrix[i])
                    # Applying the sigmoid function
                    self.ActivationMatrix[i + 1] = 1 / (1 + np.exp(-self.ActivationMatrix[i + 1]))

            # Calculating Error Matrix - going from last node to hidden layers
            # We do not calculate the error term for the bias unit or the input layer

            for i in range(len(self.NbrNeurons) - 1):

                NodeSelect = -(i + 1)

                # If last node/output node compute cost as follows
                if NodeSelect == -1:
                    self.ErrorMatrix[-1] = self.ActivationMatrix[-1] - yVal

                # Excludes error on bias term for 1 layer before output as output does not have Error term for bias unit
                #NEED TO FIX for mutliple layers
                else:
                    self.ErrorMatrix[NodeSelect] = np.dot(self.ThetaMatrix[-1][:, 1:].transpose(), self.ErrorMatrix[-1]) * (self.ActivationMatrix[NodeSelect][1:] * (1 - self.ActivationMatrix[NodeSelect][1:]))

            # Computing the Delta Terms(Gradient Accumulator)

            for i in range(len(self.NbrNeurons) - 1):

                # Exlude error term on bias unit
                if i < len(self.NbrNeurons) - 2:
                    self.GradientAcc[i] = self.GradientAcc[i] + np.dot(self.ErrorMatrix[i][1:], self.ActivationMatrix[i].transpose())

                # No Error term for bias unit on output unit
                else:
                    self.GradientAcc[i] = self.GradientAcc[i] + np.dot(self.ErrorMatrix[i], self.ActivationMatrix[i].transpose())

        # Computing the partial derivative (outside loop) and performing gradient descent
        self.PartialDerive = []
        for i in range(len(self.GradientAcc)):

            # Setting all theta bias term to zero as we dont regularize that term
            theta_adj = copy.deepcopy(self.ThetaMatrix[i])
            theta_adj[:, 0] = 0;

            # Compute partial derivative term and appending to a list
            Partial = (1 / self.m) * self.GradientAcc[i] + (lamb / self.m) * theta_adj
            self.PartialDerive.append(Partial)

        # Perform gradient descent and return the new theta matrix
        for i in range(len(self.ThetaMatrix)):
            self.ThetaMatrix[i] = self.ThetaMatrix[i] - alpha * (self.PartialDerive[i])


import numpy as np
import math
from numpy import genfromtxt


my_data = genfromtxt('diabetes.csv', delimiter=',')[1:, :]
Y = my_data[:, -1]
X = my_data[:, :-1]


Model = NeuralNetwork([8, 2, 1])

Model.Train(X, Y, 1, 0.1, 10)
