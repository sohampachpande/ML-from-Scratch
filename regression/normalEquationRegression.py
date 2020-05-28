"""
This file solves linear regression problem by normal equation method 
Y' = X*Theta
e = Y-X*Theta
differential of squared error e^2 = e^T*e would be zero
On solving we get
Theta = (X^T*X)^-1 * X^T*y
"""

import numpy as np

# all matrix inputs as np arrays


class normalEquationRegression():
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.sampleCount, self.features = X.shape[0], X.shape[1]
        self.theta = None

    def train(self):
        unitColumnMatrix = np.ones((self.sampleCount, 1))
        tempX = np.append(unitColumnMatrix, self.X, axis=1)
        t1 = np.linalg.inv(tempX.T.dot(tempX))
        t2 = t1.dot(tempX.T)
        self.theta = t2.dot(self.y)
        return self.theta

    def predict(self, XTest):
        unitColumnMatrix = np.ones((XTest.shape[0], 1))
        tempX = np.append(unitColumnMatrix, XTest, axis=1)
        return tempX.dot(self.theta)
