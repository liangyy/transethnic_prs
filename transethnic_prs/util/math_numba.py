import numpy as np
import numba as nb

@nb.jit(nb.float64(nb.float64, nb.float64))
def soft_thres_numba(x, thres):
    tmp = np.abs(x) - thres
    if tmp < 0:
        return 0
    else:
        return np.sign(x) * tmp

@nb.jit(nb.float64[::1](nb.float64[:, ::1]))
def calc_XXt_diag_numba(X):
    return np.power(X, 2).sum(axis=1)
    
@nb.jit(nb.float64[::1](nb.float64[:, ::1], nb.float64[::1]))
def calc_Xy_numba(X, y):
    return X @ y

@nb.jit(nb.float64[:, ::1](nb.float64[:, ::1]))
def mean_center_col_2d_numba(x):
    x_m = np.sum(x, axis=0) / x.shape[0]
    for i in range(x.shape[0]):
        x[i] = x[i] - x_m
    return x
    
@nb.jit(nb.float64[::1](nb.float64[::1]))
def mean_center_col_1d_numba(x):
    return x - x.mean()

@nb.jit(nb.float64[::1](nb.float64[:, ::1]))
def calc_varx_numba(mat):
    x = mean_center_col_2d_numba(mat)
    return (x ** 2).sum(axis=0) / (x.shape[0] - 1)

@nb.jit(nb.float64[:, ::1](nb.float64[:, ::1]))
def calc_covx_numba(mat):
    return np.cov(mat)
    
@nb.jit(nb.float64[::1](nb.float64[::1]))
def standardize_1d_numba(x):
    x = mean_center_col_1d_numba(x)
    denom = np.sqrt((x ** 2).mean())
    if denom == 0:
        return x
    else:
        return x / denom
    