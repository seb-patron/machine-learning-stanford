import numpy as np 
import pandas as pd 


def computeCost(X, y, theta):
    temp = np.power((theta * X.T) - y, 2)
    return np.sum(temp) / (2 * len(X))