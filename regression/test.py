import pandas as pd 
from regressionNormalEquation import normalEquationRegression
from gradientDescentRegression import gradientDescentRegression
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

	# Normal Equation Regression
	n = normalEquationRegression(X,y)
	n.train()
	# Predict
	yTestPredict = n.predict(XTest)
	# Get RMS
	predictedRMS = rootMeanSquareError(yTest, yTestPredict)
	print("NormalEquationRegression RMS Error", predictedRMS)

	# gradient Descent Regression
	gD = gradientDescentRegression(X,y,alpha=0.0000001, numberIterations=20)
	gD.train()
	# Predict
	GDyTestPredict = gD.predict(XTest)
	# Get RMS
	GDpredictedRMS = rootMeanSquareError(yTest, GDyTestPredict)
	print("gradientDescentRegression RMS Error", GDpredictedRMS)