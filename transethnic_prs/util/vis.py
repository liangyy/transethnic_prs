import matplotlib.pyplot as plt
import numpy as np

def vis_path_beta(beta_mat, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    for i in range(beta_mat.shape[0]):
        ax.plot(beta_mat[i, :], '.')
    return ax

def vis_cor(y, yp, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    cor_ = []
    for i in range(yp.shape[1]):
        cor_.append(np.corrcoef(y, yp[:, i])[1, 0])
    ax.plot(cor_, '.')
    return ax