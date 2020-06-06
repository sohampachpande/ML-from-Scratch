import numpy as np

def sigmoid(x):
	"""
	x is a nx1 vector
	"""
	return 1/(1+np.exp(-x))


class logisticRegression():
    def __init__(self, X, y, alpha=0.03, numberIterations=50):
        self.X = X
        self.y = y

        self.sampleCount, self.features = X.shape[0], X.shape[1]
        self.alpha = float(alpha)
        self.numberIterations = numberIterations
        self.theta = None

        unitColumnMatrix = np.ones((self.sampleCount, 1))
        self.augmentedX = np.append(unitColumnMatrix, self.X, axis=1)


    def initialiseTheta(self):
        self.theta = np.random.rand(self.features + 1, 1)

    def updateTheta(self):
        currentTheta = self.theta
        epsilon = self.y - sigmoid(self.augmentedX.dot(currentTheta))
        differential = self.augmentedX.T.dot(epsilon)
        return currentTheta + self.alpha * differential / self.sampleCount

    def train(self):
        self.initialiseTheta()
        for i in range(self.numberIterations):
            self.theta = self.updateTheta()
        return self.theta

    def predict(self, XTest):
        unitColumnMatrix = np.ones((XTest.shape[0], 1))
        self.augmentedX = np.append(unitColumnMatrix, XTest, axis=1)
        ans = sigmoid(self.augmentedX.dot(self.theta))
        return np.round_(ans)

    def classificationAccuracy(self, XTest, yTest):
    	yPred = self.predict(XTest)
    	counts = (yPred==yTest)
    	return counts.mean()*100