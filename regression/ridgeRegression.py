import numpy as np
from autograd import grad

from normalEquationRegression import normalEquationRegression
from gradientDescentRegression import gradientDescentRegression, gradientDescentAutogradRegression


# all matrix inputs as np arrays

class normalEquationRidgeRegression(normalEquationRegression):
    
    def __init__(self, X, y, lambda_ = 0.01):
        super(normalEquationRidgeRegression, self).__init__(X, y)
        self.lambda_ = lambda_

    def train(self):
        unitColumnMatrix = np.ones((self.sampleCount,1))
        tempX = np.append(unitColumnMatrix, self.X, axis=1)
        t1 = np.linalg.inv(tempX.T.dot(tempX) + np.absolute(self.lambda_)*(np.identity(tempX.shape[1])))
        t2 = t1.dot(tempX.T)
        self.theta = t2.dot(self.y)
        return self.theta


class gradientDescentRidgeRegression(gradientDescentRegression):
    def __init__(self, X, y, alpha, numberIterations=50, lambda_ = 0.01):
        super(gradientDescentRidgeRegression, self). __init__(X, y, alpha, numberIterations)
        self.lambda_ = lambda_

    def updateTheta(self):
        currentTheta = self.theta
        epsilon =  self.y - self.augmentedX.dot(currentTheta)  
        differential = self.augmentedX.T.dot(epsilon) - np.absolute(self.lambda_)*self.theta
        return currentTheta + self.alpha*differential/self.sampleCount


class gradientDescentAutogradRidgeRegression(gradientDescentAutogradRegression):
    def __init__(self, X, y, alpha, numberIterations=50, lambda_ = 0.01):
        super(gradientDescentAutogradRidgeRegression, self). __init__(X, y, alpha, numberIterations)
        self.lambda_ = lambda_
        self.gradientFunction = grad(self.trainingLoss)

    def trainingLoss(self, theta):
        yPredicted = self.augmentedX.dot(theta)
        epsilon = self.y - yPredicted
        mse = (np.sum(epsilon**2) + self.lambda_*np.matmul(theta.T, theta))/self.sampleCount
        return mse

    # def trainingLoss(self, theta):
    #     yPredicted = self.augmentedX.dot(theta)
    #     mse = np.sum((self.y-yPredicted)**2)/self.sampleCount
    #     return mse
