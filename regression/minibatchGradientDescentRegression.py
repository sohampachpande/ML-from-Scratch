import numpy as np
from autograd import grad

from gradientDescentRegression import gradientDescentRegression, gradientDescentAutogradRegression


class minibatchGradientDescentRegression(gradientDescentRegression):
    def __init__(self, X, y, batchSize=4, alpha=0.0000001,
                 numberIterations=50):
        super(minibatchGradientDescentRegression,
              self).__init__(X, y, alpha, numberIterations)
        self.batchSize = batchSize

    def generateBatch(self):
        indexArr = np.random.choice(self.sampleCount,
                                    size=self.batchSize,
                                    replace=False)
        y = self.y[indexArr]
        X = self.augmentedX[indexArr]
        return X, y

    def updateTheta(self):
        currentTheta = self.theta
        XBatch, yBatch = self.generateBatch()
        epsilon = yBatch - XBatch.dot(currentTheta)
        differential = XBatch.T.dot(epsilon)
        return currentTheta + self.alpha * differential.reshape((-1, 1))


class minibatchGradientDescentAutogradRegression(
        gradientDescentAutogradRegression):
    def __init__(self,
                 X,
                 y,
                 batchSize=16,
                 alpha=0.0000001,
                 numberIterations=50):
        super(minibatchGradientDescentAutogradRegression,
              self).__init__(X, y, alpha, numberIterations)
        self.batchSize = batchSize
        self.gradientFunction = grad(self.trainingLoss)

    def generateBatch(self):
        indexArr = np.random.choice(self.sampleCount,
                                    size=self.batchSize,
                                    replace=False)
        y = self.y[indexArr]
        X = self.augmentedX[indexArr]
        return X, y

    def trainingLoss(self, theta):
        XBatch, yBatch = self.generateBatch()
        yPredicted = XBatch.dot(theta)
        epsilon = yBatch - yPredicted
        mse = np.sum((epsilon)**2) / self.batchSize
        return mse

    def updateTheta(self):
        currentTheta = self.theta
        currentTheta -= self.gradientFunction(currentTheta) * self.alpha
        return currentTheta