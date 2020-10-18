import numpy as np
import operator as op


def var_covar_matrix(X, mean=None, axis=0):
    assert len(X.shape) == 2, 'must operate on a matrix of 2 dimensions'
    if axis == 1:  # calculate on transpose
        return var_covar_matrix(X.T, mean=mean)
    elif axis != 0:
        raise ValueError('axis must 0 or 1')
    # axis is now == 0

    if mean is None:
        mean = np.mean(X, axis=axis)
    diff = X - mean

    # sum of outer products for each vector divided by the number of vectors
    rv = (diff.T @ diff) / X.shape[0]
    return rv


class Transformation:
    """Abstract Base Class for transformation objects."""

    def fit(self, X, y=None):
        """fit the transformation to data X

        :param X: input data (first dimension should represent rows)
        :param y: optional - data labels
        """
        raise NotImplementedError('This is an abstract method')

    def transform(self, X, y=None):
        """transform X into the representation domain.

        Raises an exception if fit has not first been called.

        :param X: input data (first dimension should represent rows)
        :param y: optional - data labels
        """
        raise NotImplementedError('This is an abstract method')

    def inverse_transform(self, W, y=None):
        """Transform the representation back into the data domain.

        :param X: input data (first dimension should represent rows)
        :param y: optional - data labels
        """
        raise NotImplementedError('This is an abstract method')


class IdentityTransformation(Transformation):
    """IdentityTransformation. A transformation that does nothing

    useful for testing purposes
    """


    def __init__(self):
        pass

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        return X

    def inverse_transform(self, W):
        return W

    def __str__(self):
        return 'IdentityTransformation()'


class PCA(Transformation):

    def __init__(self, components=None, normalize=True):
        assert not normalize, 'this is not implemented'
        # assumes normalized data (mean of 0 over all axes, sd of 1)
        self.k = components
        self.normalize = normalize

    def fit(self, X, y=None):
        cov = var_covar_matrix(X, mean=np.zeros(X.shape[1]))
        val, vec = np.linalg.eigh(cov)  # cov is symmetric, so eigh performs better

        # vecs columns are eigenvectors
        pairs = sorted(zip(val, vec.T), key=op.itemgetter(0), reverse=True)
        self.components = np.array(list(map(op.itemgetter(1), pairs[:self.k]))).T
        self.inverse_transform_components = (
                self.components.T @ np.linalg.inv(self.components @ self.components.T))


    def transform(self, X):
        return X @ self.components


    def inverse_transform(self, W):
        return W @ self.inverse_transform_components


class NMF(Transformation):
    pass
