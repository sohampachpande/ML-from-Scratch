import pandas as pd 
from normalEquationRegression import normalEquationRegression
from gradientDescentRegression import gradientDescentRegression, gradientDescentAutogradRegression
from ridgeRegression import *
from errorUtil import rootMeanSquareError

if __name__ == '__main__':

	# Read and shuffle data
	realEstate = pd.read_csv("/home/soham/Coding/ML-from-scratch/dataset/realestate.csv", header=None, index_col=False)
	realEstate = realEstate.sample(frac=1).reset_index(drop=True)

	# Split into train and test data
	ntrain = len(realEstate)*0.75
	X = realEstate.loc[:ntrain,:5].values
	y = realEstate.loc[:ntrain,6:].values

	XTest = realEstate.loc[ntrain:,:5].values
	yTest = realEstate.loc[ntrain:,6:].values

	# # # Normal Equation Regression
	# learner = normalEquationRidgeRegression(X,y)
	# learner.train()
	# # Predict
	# yTestPredict = learner.predict(XTest)
	# # Get RMS
	# predictedRMS = rootMeanSquareError(yTest, yTestPredict)
	# print("NormalEquationRegression RMS Error", predictedRMS)

	# # # Gradient Descent Regression
	# learner = gradientDescentRegression(X,y,alpha=0.0000001, numberIterations=25)
	# learner.train()
	# # Predict
	# yTestPredict = learner.predict(XTest)
	# # Get RMS
	# predictedRMS = rootMeanSquareError(yTest, yTestPredict)
	# print("gradientDescentRegression RMS Error", predictedRMS)

	# # # gradient Descent Autograd Regression
	learner = gradientDescentAutogradRegression(X,y,alpha=0.0000001, numberIterations=25)
	learner.train()
	# Predict
	yTestPredict = learner.predict(XTest)
	# Get RMS
	predictedRMS = rootMeanSquareError(yTest, yTestPredict)
	print("gradientDescentAutogradRegression RMS Error", predictedRMS)


	# learner = normalEquationRidgeRegression(X,y, lambda_ = 0.5 ) 
	# learner.train()
	# # Predict
	# yTestPredict = learner.predict(XTest)
	# # Get RMS
	# predictedRMS = rootMeanSquareError(yTest, yTestPredict)
	# print("normalEquationRidgeRegression RMS Error", predictedRMS)

	# learner = gradientDescentRidgeRegression(X, y, alpha=0.0000001, numberIterations=25, lambda_ = 0.01 ) 
	# learner.train()
	# # Predict
	# yTestPredict = learner.predict(XTest)
	# # Get RMS
	# predictedRMS = rootMeanSquareError(yTest, yTestPredict)
	# print("gradientDescentRidgeRegression RMS Error", predictedRMS)


	learner = gradientDescentAutogradRidgeRegression(X,y,alpha=0.0000001, numberIterations=25, lambda_ = 5 ) 
	learner.train()
	# Predict
	yTestPredict = learner.predict(XTest)
	# Get RMS
	predictedRMS = rootMeanSquareError(yTest, yTestPredict)
	print("gradientDescentAutogradRidgeRegression RMS Error", predictedRMS)