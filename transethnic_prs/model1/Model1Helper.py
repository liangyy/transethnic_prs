import numpy as np
from collections import OrderedDict

def get_lambda_seq(lambda_max, nlambda, ratio_lambda):
    if isinstance(lambda_max, list):
        o = []
        for lm in lambda_max:
            lambda_min = lm / ratio_lambda
            o.append(np.exp(np.linspace(np.log(lm), np.log(lambda_min), num=nlambda)))
        return o
    elif isinstance(lambda_max, OrderedDict):
        o_dict = OrderedDict()
        for w in lambda_max.keys():
            o_dict[w] = get_lambda_seq(lambda_max[w], nlambda, ratio_lambda)
        return o_dict
    else:
        lambda_min = lambda_max / ratio_lambda
        return np.exp(np.linspace(np.log(lambda_max), np.log(lambda_min), num=nlambda))
def alpha_lambda_to_w1_w2(alpha, lambda_):
    '''
    w1 = lambda * alpha
    w2 = lambda * (1 - alpha) / 2
    '''
    return lambda_ * alpha, lambda_ * (1 - alpha) / 2
def solve_path_param_sanity_check(alpha, nlambda, ratio_lambda):
    if alpha > 1 or alpha < 0:
        raise ValueError('Only alpha in [0, 1] is acceptable.')
    if not isinstance(nlambda, int) or nlambda < 1:
        raise ValueError('nlambda needs to be integer and nlambda >= 1')
    if not isinstance(nlambda, int) or ratio_lambda <= 1:
        raise ValueError('ratio_lambda needs to be integer and ratio_lambda > 1')
def merge_list(ll):
    return np.concatenate(ll, axis=0)