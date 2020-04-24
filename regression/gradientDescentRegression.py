"""
This file solves linear regression problem by gradient Descent
"""

import numpy as np

# all matrix inputs as np arrays 

class gradientDescentRegression():
    def __init__(self, X, y, alpha, numberIterations=50):
        self.X = X
        self.y = y
        self.sampleCount, self.features = X.shape[0], X.shape[1]
        self.alpha = float(alpha)
        self.numberIterations = numberIterations
        self.theta = None

    def train(self):
        unitColumnMatrix = np.ones((self.sampleCount,1))
        tempX = np.append(unitColumnMatrix, self.X, axis=1)
        theta = np.zeros((self.features+1, 1))

        for i in range(self.numberIterations):
            epsilon =  self.y - tempX.dot(theta)  
            differential = tempX.T.dot(epsilon)
            theta = theta + self.alpha*differential/self.sampleCount

        self.theta = theta
        return theta

    def predict(self, XTest):
        unitColumnMatrix = np.ones((XTest.shape[0],1))
        tempX = np.append(unitColumnMatrix, XTest, axis=1)
        return tempX.dot(self.theta)
