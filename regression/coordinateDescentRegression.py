import numpy as np

class coordinateDescentRegression():
    def __init__(self, X, y, numberIterations=50):
        self.X = X
        self.y = y

        self.sampleCount, self.features = X.shape[0], X.shape[1]
        self.numberIterations = numberIterations
        self.theta = None

        unitColumnMatrix = np.ones((self.sampleCount,1))
        self.augmentedX = np.append(unitColumnMatrix, self.X, axis=1)


    def initialiseTheta(self):
        self.theta = np.random.rand(self.features+1, 1)

    def updateTheta(self):
        currentTheta = self.theta

        for j in range(self.features+1):
            epsilon =  self.y - self.augmentedX.dot(currentTheta)
            temp1 = epsilon + np.matmul(self.augmentedX[:,j].reshape(self.augmentedX.shape[0],1), currentTheta[j].reshape(currentTheta.shape[1],1))
            p = np.matmul(self.augmentedX[:,j].T, temp1)
            z = np.sum(np.square(self.augmentedX[:,j]))
            currentTheta[j] = p/z

        return currentTheta

    def train(self):
        self.initialiseTheta()
        for i in range(self.numberIterations):
            self.theta = self.updateTheta()
        return self.theta


    def predict(self, XTest):
        unitColumnMatrix = np.ones((XTest.shape[0],1))
        self.augmentedX = np.append(unitColumnMatrix, XTest, axis=1)
        return self.augmentedX.dot(self.theta)