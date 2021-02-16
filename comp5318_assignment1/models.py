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
        return f'GNB()'


class MultinomialLogisticRegression(TrivialModel):

    def __init__(self, classes, lr=.01, maxiter=1000):
        raise NotImplementedError()
        self._betas = None
        self._classes = classes
        self._metavalues = dict(
                loss=[],
                maxiter=maxiter,
                learning_rate=lr,
        )

    def fit(self, X, y):
        # fit the MLR model with GD
        assert len(set(y) - set(self._classes)) == 0, (f"Unknown classes found!"
        f" got {set(y) - set(self._classes)}")
        self._betas = np.random.rand(X.shape[1], len(self._classes))
        Y = self.to_onehot(y)

        for _ in range(self._metavalues['maxiter']):
            X @ self._betas
            preds = self._pred(X)
            self._metavalues['loss'].append(self._loss(Y, preds))
            grad = self._grad(X, Y)
            self._betas -= grad * self._metavalues['learning_rate']

    def _grad(self, X, Y):
        W = np.exp(-X @ self._betas)
        Wp1 = W + 1
        cats = W * (Y / Wp1 - (Y - 1) * W / (1 - 1 / Wp1) / Wp1**2)
        dims = X

        return -((cats[:,:,None] * dims[:, None, :]).sum(axis=0) / X.shape[0] / X.shape[1] / Y.shape[1]).T

    def _loss(self, Y, preds):
        clipped = np.clip(preds, 1e-15, np.inf) / len(self._classes)  # cannot have 0 or 1 (but very close)
        return np.sum(-(1-Y) * np.log(1 - clipped) - Y * np.log(clipped))

    def _pred(self, X):
        num = 1 / (1 + X @ self._betas)
        denom = num.sum(axis=1)
        return (num.T / denom).T

    def predict(self, X, y=None):
        return [self._classes[m] for m in
                np.argmax(self._pred(X), axis=1)]

    def to_onehot(self, Y):
        out = np.zeros((len(Y), len(self._classes)))
        for i, c in enumerate(self._classes):
            out[:,i] = Y == c
        return out


def cosine_distance(X, Y):
    norms = np.linalg.norm(X, axis=1)[:, None] * np.linalg.norm(Y, axis=1)[None, :]
    
    dot = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            dot[i, j] = x @ y
    return 1 - dot / norms


class KNN:

    def __init__(self, k=3, distancefunction='euclidian', weigh_voting=True):
        self.k = k
        self.distfn = {
                'euclidian': lambda x, y: np.linalg.norm(x[:, None, :] - y[None, :, :], axis=2),
                'manhatten': lambda x, y:np.sum(np.abs(x[:, None, :] - y[None, :, :]), axis=2),
                'cosine': cosine_distance,
        }[distancefunction]
        self._name = f'{type(self).__name__}(k={k}, distancefunction={distancefunction})'
        self.weigh_voting = weigh_voting

    def fit(self, x, y):
        self.x = x
        self.y = y

    def predict(self, x, batchsize=10, k=None, dists=None, return_dists=False):
        if dists is None:
            all_dists = []
            for batch in breakup(x, batchsize):

                all_dists.append(self.distfn(batch, self.x))
            all_dists = np.concatenate(all_dists)
        else:
            all_dists = dists

        if k is None:
            k = self.k

        import datetime
        idxs = np.argpartition(all_dists, k)[:,:k]  # k smallest distance indexes
        idxs_2 = np.argsort(all_dists, axis=1)[:,:k]
        if not self.weigh_voting:
            raise NotImplementedError()
            # TODO this method does not work - I must fix it
            res = np_mode(self.y[idxs], axis=(1,))
        else:
            all_dists[:, idxs]  # get k closest.
            # distances at those indexes
            distance_to = np.array([a[i] for a, i in zip(all_dists, idxs_2)])
            # true values at those indexes
            guesses = np.array([self.y[i] for i in idxs_2])
            # possible outcomes (may be a subset of range(10))
            possible = np.array(list(set(guesses.flatten())))
            # hardware; dimensions : represent:
            # 0 : data that is being predicted
            # 1 : possible labels the data could take
            # 2 : data that has known labels
            # actual values are True if that is the label for that known example
            mask = possible[None, :, None] == guesses[:, None, :]
            # mask the distances (at correct indecies) with the mask, calculating contributions
            # then sum them
            distcats = np.sum(mask * 1/distance_to[:, None, :], axis=2)
            # calculate the indexes
            predidx = distcats.argmax(axis=1)
            # collect the indexes
            res = possible[predidx]

        if return_dists:
            return all_dists, res
        else:
            return res

    def __str__(self):
        return self._name


class VotingKNN(KNN):

    def __init__(self, k_votes):
        self.k_votes = k_votes


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



if __name__ == '__main__':
    def test_logreg():
        X = np.concatenate((
            np.random.rand(100, 1) + 1,
            np.random.rand(100, 1) + 2,
            np.random.rand(100, 1) + 3,
            np.random.rand(100, 1) + 4,
            np.random.rand(100, 1) + 5,
            ), axis=0)

        Y = np.concatenate((
                np.ones(100, int) * 0,
                np.ones(100, int) * 1,
                np.ones(100, int) * 2,
                np.ones(100, int) * 3,
                np.ones(100, int) * 4,
            ), axis=0)

        # already one-hot
        #Y = np.concatenate((
        #    np.eye(5)[np.ones(100, int) * 0],
        #    np.eye(5)[np.ones(100, int) * 1],
        #    np.eye(5)[np.ones(100, int) * 2],
        #    np.eye(5)[np.ones(100, int) * 3],
        #    np.eye(5)[np.ones(100, int) * 4],
        #), axis=0)

        m = MultinomialLogisticRegression(classes=list(range(5)), lr=0.002, maxiter=int(1e4))
        m.fit(X, Y)

        modeltest(m, X, Y, X)
        from my_tools.tools import plot_confusion_matrix, confusion_matrix
        import matplotlib.pyplot as plt
        plot_confusion_matrix(confusion_matrix(Y, m.predict(X)), labels=list(range(5)))
        plt.show()
        plt.plot(m._metavalues['loss'])
        plt.show()


    def modeltest(model, trainx, trainy, testx, expectedtesty=None):
        model.fit(trainx, trainy)
        if expectedtesty is not None:
            assert list(model.predict(testx)) == list(expectedtesty)
        else:
            return model.predict(testx)
    test_logreg()
