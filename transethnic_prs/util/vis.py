import matplotlib.pyplot as plt
import numpy as np

def vis_path_beta(beta_mat, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    for i in range(beta_mat.shape[0]):
        ax.plot(beta_mat[i, :], '.')
    return ax

def corrcoef(y1, y2):
    y1 = y1 - y1.mean()
    y2 = y2 - y2.mean()
    v1 = (y1 ** 2).sum()
    v2 = (y2 ** 2).sum()
    denom = v1 * v2
    if denom == 0:
        return np.nan
    else:
        v3 = (y1 * y2).sum()
        return v3 / np.sqrt(denom)

def vis_cor(y, yp, ax=None, return_only=False):
    if ax is None:
        _, ax = plt.subplots()
    cor_ = []
    for i in range(yp.shape[1]):
        cor_.append(corrcoef(y, yp[:, i]))
    if return_only is True:
        return cor_
    else:
        ax.plot(cor_, '.')    
        return ax