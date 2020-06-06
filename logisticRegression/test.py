import pandas as pd
from logistic import logisticRegression
import numpy as np

if __name__ == '__main__':

    # Read and shuffle data
    bank = pd.read_csv(
        "/home/soham/Coding/ML-from-scratch/dataset/PimaIndiansDiabetesDataset.csv",
        header=None,
        index_col=False)

    bank = bank.sample(frac=1).reset_index(drop=True)

    # Split into train and test data
    ntrain = len(bank) * 0.75
    X = bank.loc[:ntrain, :7].values
    y = bank.loc[:ntrain, 8:].values

    XTest = bank.loc[ntrain:, :7].values
    yTest = bank.loc[ntrain:, 8:].values

    # # # Binary Logistic Regression
    learner = logisticRegression(X,y, alpha=0.3, numberIterations=200)
    learner.train()
    # # Predict
    yTestPredict = learner.predict(XTest)
    # # Get Classification Accuracy
    predictedClassificationAccuracy = learner.classificationAccuracy(XTest, yTest)
    print("Logistic Regression Accuracy", predictedClassificationAccuracy)