# Creating a Neural Network training algorithm
import numpy as np
import copy
import math


class NeuralNetwork():

    # Defining model architecture
    def __init__(self, NbrNeurons):

        self.ThetaCount = len(NbrNeurons) - 1
        self.NbrNeurons = NbrNeurons
        self.ThetaSize = []
        self.ThetaMatrix = []

        # Generating Theta  size list
        self.ThetaSize = [(self.NbrNeurons[i + 1], self.NbrNeurons[i] + 1) for i in range(self.ThetaCount)]

        # Creating a list of theta matrices and intializing them
        self.ThetaMatrix = [(np.random.random(self.ThetaSize[i]) * 2 * 0.12 - 0.12) for i in range(len(self.ThetaSize))]

        # Creating the Activation Node Matrix including the bias term
        self.ActivationMatrix = [np.zeros((i + 1, 1)) for i in NbrNeurons]
        # If last layer then no bias term
        self.ActivationMatrix[-1] = np.zeros((NbrNeurons[-1], 1))

        # Node Error Matrix - DeepCopy as same dimensions as Activation Matrix
        self.ErrorMatrix = copy.deepcopy(self.ActivationMatrix)
        # Deleting the first error matrix as we dont have an error term for the input
        del self.ErrorMatrix[0]

        # Gradient Accumulator - same dimensions as theta matrix
        self.GradientAcc = copy.deepcopy(self.ThetaMatrix)

        # Pre-loading the bias term = 1
        for i in range(len(self.ActivationMatrix) - 1):
            self.ActivationMatrix[i][0] = 1

    # Training the model and performing backpropagation
    def Train(self, XVal, yVal, lamb, alpha, iters):

        # Nbr Training Examples
        self.m = XVal.shape[0]

        # Setting the X_Val as the first Activation Layer - Note the first term is a bias
        self.ActivationMatrix[0][1:] = XVal

        # FeedFoward Network - One observation at a time
        for i in range(len(self.NbrNeurons) - 1):

            # For the last matrix/output matrix there is no bias term
            # Hence we don't need to only the second element onwards (no bias in output)

            if i < len(self.NbrNeurons) - 2:

                self.ActivationMatrix[i + 1][1:] = self.ThetaMatrix[i] @ self.ActivationMatrix[i]
                # Applying the sigmoid function
                self.ActivationMatrix[i + 1][1:] = 1 / (1 + np.exp(-self.ActivationMatrix[i + 1][1:]))

            # If output node we dont have a bias term
            elif i == len(self.NbrNeurons) - 2:
                self.ActivationMatrix[i + 1] = self.ThetaMatrix[i] @ self.ActivationMatrix[i]
                # Applying the sigmoid function
                self.ActivationMatrix[i + 1] = 1 / (1 + np.exp(-self.ActivationMatrix[i + 1]))

            print(i)

        # Calculating Error Matrix - going from last node to hidden layers
        # We do not calculate the error term for the bias unit or the input layer

        for i in range(len(self.NbrNeurons) - 1):

            NodeSelect = -(i + 1)

            # If last node/output node compute cost as follows
            if NodeSelect == -1:
                self.ErrorMatrix[-1] = self.ActivationMatrix[-1] - yVal

            # Excludes error on bias term for 1 layer before output as output does not have Error term for bias unit
            elif NodeSelect == -2:
                self.ErrorMatrix[NodeSelect][1:] = (self.ThetaMatrix[NodeSelect + 1][:, 1:] .transpose() @ self.ErrorMatrix[NodeSelect + 1]) * (self.ActivationMatrix[NodeSelect][1:] * (1 - self.ActivationMatrix[NodeSelect][1:]))

            # Excludes error on bias term for 2 layers + before output node
            else:
                self.ErrorMatrix[NodeSelect][1:] = (self.ThetaMatrix[NodeSelect + 1][:, 1:] .transpose() @ self.ErrorMatrix[NodeSelect + 1][1:]) * (self.ActivationMatrix[NodeSelect][1:] * (1 - self.ActivationMatrix[NodeSelect][1:]))

        # Computing the Delta Terms(Gradient Accumulator)

        for i in range(len(self.NbrNeurons) - 1):

            # No Error term for bias unit on output unit
            if i < len(self.NbrNeurons):
                self.GradientAcc[i] = self.GradientAcc[i] + (self.ErrorMatrix[i][1:] @ self.ActivationMatrix[i].transpose())

            else:
                self.GradientAcc[i] = self.GradientAcc[i] + (self.ErrorMatrix[i][1:] @ self.ActivationMatrix[i].transpose())

        # Computing the partial derivative (outside loop) and performing gradient descent


Model = NeuralNetwork([3, 2, 1])
XVal = np.array([[0], [0], [0]])
yVal = np.array([[1]])


Model.Train(XVal, yVal, 0, 0, 0)
print(Model.GradientAcc[-2])
