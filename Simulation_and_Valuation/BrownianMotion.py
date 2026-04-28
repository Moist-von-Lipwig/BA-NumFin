import numpy as np
import math as mth

def Forward_Method(X, T):
    # X is a mxn-dimensional normal distributed vector and T is a list of n time-steps
    n = np.size(T)
    W = np.zeros(np.shape(X))
    W[0, :] = mth.sqrt(T[0]) * X[0, :]
    for i in range(1, n):
        W[i, :] = W[i-1, :] + mth.sqrt(T[i]-T[i-1]) * X[i, :]
    W = np.append(np.array([np.zeros(np.size(X[0,:]))]), W, axis=0)
    return W.T #Transposing such that one row is one path

#def Brownian_Bridge