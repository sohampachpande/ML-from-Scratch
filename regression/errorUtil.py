# Utility functions for testing


# the inputs are numpy arrays of size (n,1)
def rootMeanSquareError(groundTruth, predicted):
    from sklearn.metrics import mean_squared_error
    from math import sqrt

    rms = sqrt(mean_squared_error(groundTruth, predicted))

    return rms