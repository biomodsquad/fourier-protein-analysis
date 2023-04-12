# function K (kernel function)  when kernel = rbf
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def KN(X,gamma) :
    d = euclidean_distances(X, X)
    K = np.exp(-gamma*d**2)
    return K