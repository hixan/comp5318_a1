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

    def fit(self, x, y):
        levels = set(y)
        self.dists = {}
        self.pclasses = {}
        for level in levels:
            xcat = x[y==level]
            self.dists[level] = dict(mu=xcat.mean(axis=0), sigma=xcat.std(axis=0))
            self.pclasses[level] = len(xcat) / len(x)

    def norm_cdf(self, x, mu, sigma):
        """norm_cdf.

        :param x: ndarray (n by nclasses) data
        :param mu: ndarray (nclasses) means of each features
        :param sigma: ndarray (nclasses) sd of each feature
        :return: ndarray (n by nclasses) probability of each feature being that value
        """
        rv = np.exp(-(x-mu)**2 / 2 / sigma ** 2) / np.sqrt(2 * np.pi * sigma ** 2)
        return rv

    def p_conditional(self, x, c):
        rv = self.norm_cdf(x, **self.dists[c])
        return rv

    def predict(self, x):
        classes = np.array(list(self.pclasses.keys()))
        probs = np.log(np.array([self.p_conditional(x, c) * self.pclasses[c] for c in classes])).sum(axis=2)
        return classes[np.argmax(probs, axis=0)]



