import numpy as np
import math
from numpy import genfromtxt


my_data = genfromtxt('diabetes.csv', delimiter=',')[1:, :]
Y = my_data[:, -1]
X = my_data[:, :-1]

NbrNeurons = [3, 2, 6, 2]

ThetaMatrix = [np.random.random((NbrNeurons[i + 1], NbrNeurons[i] + 1)) for i in range(len(NbrNeurons) - 1)]

ActivationMatrix = [np.zeros((NbrNeurons[i] + 1, 1)) if i <= (len(NbrNeurons) - 2) else np.zeros((NbrNeurons[i], 1)) for i in range(len(NbrNeurons))]

for i in range(len(ActivationMatrix) - 1):
    ActivationMatrix[i][0] = 1

ErrorMatrix = [np.zeros((NbrNeurons[i + 1], 1)) for i in range(len(NbrNeurons) - 1)]
