import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.decomposition import NMF
from operator import itemgetter

class TrivialModel:

    def __init__(self, rv=0):
        self.rv = rv

    def fit(self, x, y):
        pass

    def predict(self, x):
        return [self.rv]*len(x)

class GNB:

    def __init__(self, sigma_adjust=1e-4):
        self.sigma_adjust = sigma_adjust

    def fit(self, x, y):
        self.levels = sorted(list(set(y)))
        self.means = []
        self.sds = []
        self.pclasses = []
        for i, level in enumerate(sorted(self.levels)):
            xcat = x[y==level]
            self.means.append(np.mean(xcat, axis=0))
            self.sds.append(np.std(xcat, axis=0))
            self.pclasses.append(len(xcat) / len(x))
        self.means = np.array(self.means)
        self.sds = np.array(self.sds)
        self.pclasses = np.array(self.pclasses)

    def norm_cdf(self, x, mu, sigma):
        """norm_cdf.

        :param x: ndarray (n by k) data
        :param mu: ndarray (k by c) means of each feature for each class
        :param sigma: ndarray (k by c) sd of each feature for each class
        :return: ndarray (n by k by c) probability contribution of each feature being that value.
        """

        n, k = x.shape
        c = mu.shape[1]

        # sanity checks
        assert mu.shape[0] == k
        assert mu.shape == sigma.shape

        xt = np.transpose(np.tile(x, (c,1,1)), (1, 2, 0))
        mut = np.tile(mu, (n, 1, 1))
        sigmat = np.tile(sigma, (n, 1, 1)) + self.sigma_adjust

        # the above are now in the same shapes, meaning that *, +, -, /, **
        # all operate element-wise in a predictable way
        assert xt.shape == (n, k, c), f'{xt.shape} =/= {(n, k, c)}'
        assert mut.shape == (n, k, c), f'{mut.shape} =/= {(n, k, c)}'
        assert sigmat.shape == (n, k, c), f'{sigmat.shape} =/= {(n, k, c)}'

        inexp = -(xt - mut)**2 / (2 * sigmat ** 2)
        num = np.exp(inexp)
        den = np.sqrt(2 * np.pi * sigmat**2)

        rv =  num / den
        # sigmat can sometimes be 0 with homogenous features (features that
        # contribute nothing) This means that p(ci | xi) = 0 as xi xc for all ci.
        return rv

    def predict(self, x):
        # tile and make of the form (n x k x c)
        ind_probs = self.norm_cdf(x, self.means.T, self.sds.T)
        conditionalprobs = np.sum(np.log(ind_probs), axis=1)

        # pclasses has shape (c,), so it is repeated for each row in conditionalprobs * pclasses

        return np.argmax(conditionalprobs + np.log(self.pclasses), axis=1)

    def __str__(self):
        return f'GNB({self.sigma_adjust})'




