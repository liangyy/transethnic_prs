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