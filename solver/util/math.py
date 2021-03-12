import numpy as np

def _get_positive_side(v):
    if v < 0:
        return 0
    else:
        return v
        
def soft_thres(a, b):
    '''
    sign(a) * (abs(a) - b)_+
    '''
    if b < 0:
        raise TypeError('b needs to be positive number.')
    if isinstance(a, float) and isinstance(b, float):
        pass
    else:
        raise TypeError('soft_thres takes float as input.')

    return np.sign(a) * _get_positive_side(abs(a) - b)

def l2_norm_sq(vec):
    if len(vec.shape) > 1:
        raise ValueError('l2_norm_sq takes 1-dim np.array.')
    return (vec ** 2).sum()

def l1_norm(vec):
    if len(vec.shape) > 1:
        raise ValueError('l1_norm takes 1-dim np.array.')
    return np.absolute(vec).sum()
    