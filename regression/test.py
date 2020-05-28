import pandas as pd 
from normalEquationRegression import normalEquationRegression
from gradientDescentRegression import gradientDescentRegression, gradientDescentAutogradRegression
from ridgeRegression import *
from coordinateDescentRegression import coordinateDescentRegression
from lassoRegression import coordinateDescentLassoRegression, gradientDescentAutogradLassoRegression
from stochasticGradientDescentRegression import stochasticGradientDescentRegression, stochasticGradientDescentAutogradRegression
from minibatchGradientDescentRegression import minibatchGradientDescentRegression, minibatchGradientDescentAutogradRegression
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
	# learner = gradientDescentAutogradRegression(X,y,alpha=0.0000001, numberIterations=25)
	# learner.train()
	# # Predict
	# yTestPredict = learner.predict(XTest)
	# # Get RMS
	# predictedRMS = rootMeanSquareError(yTest, yTestPredict)
	# print("gradientDescentAutogradRegression RMS Error", predictedRMS)


	# # # gradient Descent Autograd Regression
	# learner = gradientDescentAutogradRegression(X,y,alpha=0.0000001, numberIterations=25)
	# learner.train()
	# # Predict
	# yTestPredict = learner.predict(XTest)
	# # Get RMS
	# predictedRMS = rootMeanSquareError(yTest, yTestPredict)
	# print("gradientDescentAutogradRegression RMS Error", predictedRMS)


	# # # normalEquationRidgeRegression
	# learner = normalEquationRidgeRegression(X,y, lambda_ = 0.5 ) 
	# learner.train()
	# # Predict
	# yTestPredict = learner.predict(XTest)
	# # Get RMS
	# predictedRMS = rootMeanSquareError(yTest, yTestPredict)
	# print("normalEquationRidgeRegression RMS Error", predictedRMS)


	# # # gradientDescentRidgeRegression
	# learner = gradientDescentRidgeRegression(X, y, alpha=0.0000001, numberIterations=25, lambda_ = 0.01 ) 
	# learner.train()
	# # Predict
	# yTestPredict = learner.predict(XTest)
	# # Get RMS
	# predictedRMS = rootMeanSquareError(yTest, yTestPredict)
	# print("gradientDescentRidgeRegression RMS Error", predictedRMS)


	# # # gradientDescentAutogradRidgeRegression
	# learner = gradientDescentAutogradRidgeRegression(X,y,alpha=0.0000001, numberIterations=25, lambda_ = 5 ) 
	# learner.train()
	# # Predict
	# yTestPredict = learner.predict(XTest)
	# # Get RMS
	# predictedRMS = rootMeanSquareError(yTest, yTestPredict)
	# print("gradientDescentAutogradRidgeRegression RMS Error", predictedRMS)


	# # # coordinateDescentRegression
	# learner = coordinateDescentRegression(X,y,numberIterations=25) 
	# learner.train()
	# # Predict
	# yTestPredict = learner.predict(XTest)
	# # Get RMS
	# predictedRMS = rootMeanSquareError(yTest, yTestPredict)
	# print("coordinateDescentRegression RMS Error", predictedRMS)


	# # # coordinateDescentLassoRegression
	# learner = coordinateDescentLassoRegression(X,y,lambda_=2,numberIterations=25) 
	# learner.train()
	# # Predict
	# yTestPredict = learner.predict(XTest)
	# # Get RMS
	# predictedRMS = rootMeanSquareError(yTest, yTestPredict)
	# print("coordinateDescentLassoRegression RMS Error", predictedRMS)


	# # # gradientDescentAutogradLassoRegression
	# learner = gradientDescentAutogradLassoRegression(X,y,alpha=0.001, numberIterations=10, lambda_ = 5 ) 
	# learner.train()
	# # Predict
	# yTestPredict = learner.predict(XTest)
	# # Get RMS
	# predictedRMS = rootMeanSquareError(yTest, yTestPredict)
	# print("gradientDescentAutogradLassoRegression RMS Error", predictedRMS)


	# # # stochasticGradientDescentRegression
	# learner = stochasticGradientDescentRegression(X,y,alpha=0.0000001, numberIterations=100) 
	# learner.train()
	# # Predict
	# yTestPredict = learner.predict(XTest)
	# # Get RMS
	# predictedRMS = rootMeanSquareError(yTest, yTestPredict)
	# print("stochasticGradientDescentRegression RMS Error", predictedRMS)

	# # # stochasticGradientDescentAutogradRegression
	# learner = stochasticGradientDescentAutogradRegression(X,y,alpha=0.0000001, numberIterations=100) 
	# learner.train()
	# # Predict
	# yTestPredict = learner.predict(XTest)
	# # Get RMS
	# predictedRMS = rootMeanSquareError(yTest, yTestPredict)
	# print("stochasticGradientDescentAutogradRegression RMS Error", predictedRMS)


	# # # minibatchGradientDescentRegression
	# learner = minibatchGradientDescentRegression(X,y,batchSize=16, alpha=0.00000001, numberIterations=100) 
	# learner.train()
	# # Predict
	# yTestPredict = learner.predict(XTest)
	# # Get RMS
	# predictedRMS = rootMeanSquareError(yTest, yTestPredict)
	# print("minibatchGradientDescentRegression RMS Error", predictedRMS)
	
	# # # minibatchGradientDescentAutogradRegression
	learner = minibatchGradientDescentAutogradRegression(X,y,batchSize=16, alpha=0.00000001, numberIterations=100) 
	learner.train()
	# Predict
	yTestPredict = learner.predict(XTest)
	# Get RMS
	predictedRMS = rootMeanSquareError(yTest, yTestPredict)
	print("minibatchGradientDescentAutogradRegression RMS Error", predictedRMS)