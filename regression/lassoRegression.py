import numpy as np
from autograd import grad, elementwise_grad

from coordinateDescentRegression import coordinateDescentRegression
from gradientDescentRegression import gradientDescentAutogradRegression

class coordinateDescentLassoRegression(coordinateDescentRegression):
    def __init__(self, X, y, lambda_=0.01 , numberIterations=50):
        super(coordinateDescentLassoRegression, self).__init__(X, y, numberIterations)
        self.lambda_ = lambda_

    def updateTheta(self):
        currentTheta = self.theta
        halfLambdaSquare = 0.5*(self.lambda_)**2
        for j in range(self.features+1):
            epsilon =  self.y - self.augmentedX.dot(currentTheta)
            temp1 = epsilon + np.matmul(self.augmentedX[:,j].reshape(self.augmentedX.shape[0],1), currentTheta[j].reshape(currentTheta.shape[1],1))
            p = np.matmul(self.augmentedX[:,j].T, temp1)
            z = np.sum(np.square(self.augmentedX[:,j]))
            
            if p>halfLambdaSquare:
                currentTheta[j] = (p-halfLambdaSquare)/z
            elif p<halfLambdaSquare:
                currentTheta[j] = (p+halfLambdaSquare)/z
            else:
                currentTheta[j] = 0

        return currentTheta


class gradientDescentAutogradLassoRegression(gradientDescentAutogradRegression):
    def __init__(self, X, y, alpha=0.0000001, numberIterations=50, lambda_ = 0.01):
        super(gradientDescentAutogradLassoRegression, self).__init__(X, y, alpha, numberIterations)
        self.lambda_ = lambda_
        self.gradientFunction = elementwise_grad(self.trainingLoss)

    def trainingLoss(self, theta):
        yPredicted = self.augmentedX.dot(theta)
        epsilon = self.y - yPredicted
        mse = (np.sum(epsilon**2) + self.lambda_*np.abs(theta))/self.sampleCount
        return mse  