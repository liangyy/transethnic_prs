import numpy as np
import scipy.stats

from transethnic_prs.util.math import mean_center_col

def fast_simple_linear_regression(X, y):
    '''
    Center y and X
    Solve y ~ X[:, i] for each i
    Return bhat[i] and se[i]
    Formula:
        bhat = (x'y) / (x'x)
        se = sqrt( sigma2 / x'x )  
        sigma2 = r'r / (n - 2)
        # df = 2 (1. bhat; 2. center y and X)
        r = y - x bhat
    '''
    X_ = mean_center_col(X)
    y_ = mean_center_col(y)
    xtx = (X_ ** 2).sum(axis=0)
    xty = (X_ * y_[:, np.newaxis]).sum(axis=0)
    bhat = xty / xtx
    r = y_[:, np.newaxis] - X_ * bhat[np.newaxis, :]
    rtr = (r ** 2).sum(axis=0)
    sigma2 = rtr / (X_.shape[0] - 2)
    se = np.sqrt(sigma2 / xtx)
    return bhat, se

def z2p(z):
    '''
    Two sided p-value
    '''
    return np.exp(scipy.stats.norm.logsf(np.abs(z))) * 2
    