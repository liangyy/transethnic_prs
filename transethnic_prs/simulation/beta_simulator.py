import numpy as np

class BetaSimulator:
    def __init__(self, p):
        if isinstance(p, int):
            self.p = p
        else:
            raise TypeError('p needs to be integer.')
    def _spike_and_slab_general(self, pi0, func):
        '''
        def func(p):
            return vec # of size p
        beta_j \sim pi0 * delta0 + (1 - pi0) * func(.)
        '''
        beta = np.zeros(self.p)
        non_zero = np.random.rand(self.p) > pi0
        beta[non_zero] = func(non_zero.sum())
        return beta
    def spike_and_slab(self, pi0, sigma2):
        func = lambda x: np.random.normal(scale=np.sqrt(sigma2), loc=0, size=x)
        return self._spike_and_slab_general(pi0, func)