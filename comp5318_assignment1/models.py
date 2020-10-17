import numpy as np
from itertools import count

class TrivialModel:

    def __init__(self, rv=0):
        self.rv = rv

    def fit(self, x, y):
        pass

    def predict(self, x):
        return [self.rv]*len(x)

    def __str__(self):
        return f'TrivialModel({self.rv})'


class GNB(TrivialModel):

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
        #>>> assert xt.shape == (n, k, c), f'{xt.shape} =/= {(n, k, c)}'
        #>>> assert mut.shape == (n, k, c), f'{mut.shape} =/= {(n, k, c)}'
        #>>> assert sigmat.shape == (n, k, c), f'{sigmat.shape} =/= {(n, k, c)}'

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

def cpdiff(x, y):
    '''cartesian product difference between x, y.

    Calculates cartesian product difference between all values of x and all values of y

    Output is of the form:
    x[i] - y[j] = result[i, j]
    '''
    return x[:, None, :] - y[None, :, :]


class KNN:
    def __init__(self, k=3, distancefunction='euclidian'):
        self.k = k
        self.distfn = {
            'euclidian': lambda x, y: np.linalg.norm(cpdiff(x, y), axis=2),
            'manhatten': lambda x, y:np.sum(cpdiff(x, y), axis=2)
        }[distancefunction]
        self._name = f'{type(self).__name__}(k={k}, distancefunction={distancefunction})'

    def fit(self, x, y):
        self.x = x
        self.y = y

    def predict(self, x, batchsize=30):
        res = []
        for batch in breakup(x, batchsize):

            dists = self.distfn(batch, self.x)
            idxs = np.argpartition(dists, self.k)[:,:self.k]
            res.append(np_mode(self.y[idxs], axis=(1,)))
        return np.concatenate(res)

    def __str__(self):
        return self._name


def np_mode(a, axis=None):
    """np_mode - numpy.mode implementation (it is not in base numpy).

    Works in the same way as np.mean, but returns the mode instead.

    :param ndarray: array to perform mode on
    :param axis: axis (or axes) over which to perform the mode.
    :return: ndarray of modes

    if a.shape == (w, x, y, z) and axis == (1, 2), then the returned value
    will have the shape (w, z).
    """
    if axis is not None:
        try:
            #axis = tuple(set(range(len(a.shape))) - set(axis))
            axis = tuple(axis)
        except TypeError:
            return np_mode(a, axis=(axis,))
    options = np.array([np.sum(a == i, axis=axis)
        for i in range(np.max((a)))
        ])
    return np.argmax(options, axis=0)


def breakup(itr, batchsize):
    """break up an iterable into more managable batches

    :param itr: iterable to break up
    :param batchsize: size of batch to break up into

    This will generate batches (of length batchsize) of iterable.
    """
    for batch in count():
        batch *= batchsize
        n = batch + batchsize
        if n < len(itr):
            yield itr[batch:n]
        else:
            break
    yield itr[batch:]

