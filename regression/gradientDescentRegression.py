"""
This file solves linear regression problem by gradient Descent
"""

import autograd.numpy as np
from autograd import grad

# all matrix inputs as np arrays 

class gradientDescentRegression():
    def __init__(self, X, y, alpha, numberIterations=50):
        self.X = X
        self.y = y

        self.sampleCount, self.features = X.shape[0], X.shape[1]
        self.alpha = float(alpha)
        self.numberIterations = numberIterations
        self.theta = None

        unitColumnMatrix = np.ones((self.sampleCount,1))
        self.augmentedX = np.append(unitColumnMatrix, self.X, axis=1)


    def initialiseTheta(self):
        self.theta = np.random.rand(self.features+1, 1)

    def updateTheta(self):
        currentTheta = self.theta
        epsilon =  self.y - self.augmentedX.dot(currentTheta)  
        differential = self.augmentedX.T.dot(epsilon)
        return currentTheta + self.alpha*differential/self.sampleCount


    def train(self):
        self.initialiseTheta()
        for i in range(self.numberIterations):
            self.theta = self.updateTheta()
        return self.theta


    def predict(self, XTest):
        unitColumnMatrix = np.ones((XTest.shape[0],1))
        self.augmentedX = np.append(unitColumnMatrix, XTest, axis=1)
        return self.augmentedX.dot(self.theta)


# Gradient Descent using autograd to compute gradients
class gradientDescentAutogradRegression(gradientDescentRegression):

    def __init__(self,  X, Y, alpha, numberIterations=50):
        super(gradientDescentAutogradRegression, self).__init__(X, Y, alpha, numberIterations)
        self.gradientFunction = grad(self.trainingLoss)

    def trainingLoss(self, theta):
        yPredicted = self.augmentedX.dot(theta)
        epsilon = self.y - yPredicted
        # mse = np.matmul(epsilon.T, epsilon)/self.sampleCount
        mse = np.sum((epsilon)**2)/self.sampleCount
        return mse

    def updateTheta(self):
        currentTheta = self.theta
        currentTheta -= self.gradientFunction(currentTheta)*self.alpha
        return currentTheta