import pandas as pd 
from dtree import dTree

if __name__ == '__main__':

	# Read and shuffle data
	iris = pd.read_csv("/home/soham/Coding/ML-from-scratch/dataset/iris_data.csv", header=None, index_col = False )
	iris = iris.sample(frac=1).reset_index(drop=True)

	# Split into train and test data
	irisTrain = iris.loc[:0.7*len(iris),]
	irisTest = iris.loc[0.7*len(iris):,:4-1].reset_index(drop=True)
	irisTestTruth = iris.loc[0.7*len(iris):, 4:].rename(columns = {4:0}).reset_index(drop=True)

	# initialise CART decision tree and build it 
	tree = dTree(irisTrain,5,5)
	tree.buildTree()

	# Predict
	irisTestPredicted = tree.predict(irisTest)

	# Get Accuracy
	predictedAccuracy = tree.predictAccuracy(irisTest, irisTestTruth)
	print("predicted Accuracy", predictedAccuracy)