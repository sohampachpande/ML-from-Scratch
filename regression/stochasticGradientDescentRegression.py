import numpy as np
from autograd import grad

from gradientDescentRegression import gradientDescentRegression, gradientDescentAutogradRegression


class stochasticGradientDescentRegression(gradientDescentRegression):
    def __init__(self, X, y, alpha=0.0000001, numberIterations=50):
        super(stochasticGradientDescentRegression,
              self).__init__(X, y, alpha, numberIterations)

    def updateTheta(self):
        currentTheta = self.theta
        j = np.random.randint(self.sampleCount)
        epsilon = self.y[j] - self.augmentedX[j].dot(currentTheta)
        differential = self.augmentedX[j].T.reshape((-1, 1)).dot(epsilon)
        return currentTheta + self.alpha * differential.reshape((-1, 1))


class stochasticGradientDescentAutogradRegression(
        gradientDescentAutogradRegression):
    def __init__(self, X, y, alpha=0.0000001, numberIterations=50):
        super(stochasticGradientDescentAutogradRegression,
              self).__init__(X, y, alpha, numberIterations)
        self.gradientFunction = grad(self.trainingLoss)

    def trainingLoss(self, theta):
        j = np.random.randint(self.sampleCount)
        yPredicted = self.augmentedX[j].dot(theta)
        epsilon = self.y[j] - yPredicted
        return epsilon**2

    def updateTheta(self):
        currentTheta = self.theta
        currentTheta -= self.gradientFunction(currentTheta) * self.alpha
        return currentTheta