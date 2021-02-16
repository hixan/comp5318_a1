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
        self.normalize = normalize
        # assumes normalized data (mean of 0 over all axes, sd of 1)
        self.k = components
        self.normalize = normalize
        self._metavalues = dict(
                variances = None
                )

    def _norm(self, X):
        return (X - self.means) / self.sds

    def _inv_norm(self, W):
        return (W * self.sds) + self.means

    def fit(self, X, y=None):

        if self.k is None:
            self.k = X.shape[1]
        if self.normalize:
            self.means = np.mean(X, axis=0)
            self.sds = np.std(X, axis=0)
        else:
            self.means = np.zeros(X.shape[1])
            self.sds = np.ones(X.shape[1])

        X = self._norm(X)

        cov = var_covar_matrix(X, mean=np.zeros(X.shape[1]))
        val, vec = np.linalg.eigh(cov)  # cov is symmetric, so eigh performs better

        # vecs columns are eigenvectors
        pairs = sorted(zip(val, vec.T), key=op.itemgetter(0), reverse=True)
        self._metavalues['variances'] = np.array(list(map(op.itemgetter(0), pairs)))
        self.components = np.array(list(map(op.itemgetter(1), pairs[:self.k]))).T
        self.inverse_transform_components = self.components.T

    def transform(self, X):
        return self._norm(X) @ self.components

    def inverse_transform(self, W):
        return self._inv_norm(W @ self.inverse_transform_components)

    def __str__(self):
        return f'PCA({self.k}, {self.normalize})'


class NMF:

    def __init__(self, components, stop_threshold=0.01, max_iter=200, initial_dictionary=None, image_shape=None):
        if initial_dictionary is not None:
            initial_dictionary = initial_dictionary.copy()
        self._metavalues = dict(
            name='L2 Norm NMF',
            training_loss=[],
            training_residue=[],
            components=components,
            stop_threshold=stop_threshold,
            max_iter=max_iter,
            initial_dictionary=initial_dictionary,
            image_shape=image_shape,
        )
        self._dictionary = None
        self._inverse_dictionary = None

    def fit(self, X: np.ndarray, initial_representation=None):
        """ Assumes first dimension of X represents rows of data """

        if self._metavalues['image_shape'] is None:
            # initialise default image shape if was not previously assigned
            self._metavalues['image_shape'] = X.shape[1:]
        else:
            # sanity checks
            assert X.shape[1:] == self._metavalues['image_shape'], ('input data does '
                                                                    'not match expected shape')

        # reshape the data to be vectors instead of images (if not already reshaped)
        n: int = X.shape[0]
        p: int = np.product(X.shape[1:])
        k: int = self._metavalues['components']
        X: np.ndarray = NMF._reshape_forward(X)
        assert X.shape == (p, n)

        # n - number of input images
        # p - dimensionality of population space
        # k - number of components

        # X shape (p, n)
        # D shape (p, k)
        # R shape (k, n)

        # initialise the learning dictionary if not already initialised
        if self._metavalues['initial_dictionary'] is None:
            self._metavalues['initial_dictionary'] = np.random.rand(p, k)
        else:
            assert self._metavalues['initial_dictionary'].shape == (p, k)

        # initialize dictionary if not already done.
        if self._dictionary is None:
            self._dictionary = self._metavalues['initial_dictionary'].copy()

        # initialize representation
        if initial_representation is None:
            R: np.ndarray = np.random.rand(k, n)
        else:
            R: np.ndarray = initial_representation.copy()
            assert R.shape == (k, n)

        D: np.ndarray = self._dictionary  # alias for readability.

        # toggle optimizing between D and R
        # start with updating 'R'
        optim = 'R'

        # marker for different calls
        self._metavalues['training_loss'].append(None)
        self._metavalues['training_residue'].append(None)

        # fit the data
        for iteration in range(self._metavalues['max_iter'] * 2):  # *2 to account for alternation
            # this section follows section 2.7 of the accompanied documentation in
            # ../papers/Robust Nonnegative Matrix Factorization using L21 Norm 2011.pdf

            # only collect the loss after D has been updated
            if optim == 'D':
                diffs = X - D @ R
                loss = l2_norm(diffs)
                residue = np.linalg.norm(diffs)

                # keep these for later
                self._metavalues['training_loss'].append(loss)
                self._metavalues['training_residue'].append(residue)

                # computing if stopping condition is met
                if iteration >= 2:
                    previous_loss = self._metavalues['training_loss'][-1]
                    current_loss = self._metavalues['training_loss'][-2]
                    relative_improvement= - (previous_loss - current_loss) / previous_loss
                    if relative_improvement < self._metavalues['stop_threshold']:
                        optim = 'stop'

            if optim == 'D':
                optim = 'R'  # toggle for next time
                D *= (X @ R.T) / (D @ R @ R.T)
            elif optim == 'R':
                optim = 'D'
                R *= (D.T @ X) / (D.T @ D @ R)

            elif optim == 'stop':
                break
            else:
                assert 0, 'optim not recognised'

        self._inverse_dictionary = np.linalg.inv(D.T @ D) @ D.T

    def transform(self, X):
        """ Transform X into its representation

        :param X: row matrix/tensor of same shape as training time representing
            data. If there are n images of size 10x5, X should be of shape
            (n, 10, 5) or (n, 50) (depending on what was passed at training
            time)
        :return: row matrix (n, k) representing X.

        Returns a row oriented matrix of representation vectors of X
        """
        return (self._inverse_dictionary @ NMF._reshape_forward(X)).T

    def inverse_transform(self, R):
        """ Transform representations of X back into X.

        :param R: row oriented matrix of representation vectors
        :return: row matrix/tensor of the same shape as input (barring first
        dimension)
        """
        return self._reshape_backward(self._dictionary @ R.T)

    def get_metavalues(self):
        """ NMF.get_metavalues

        returns a dict with the following attributes:
        'name' : name of the algorithm
        'training_loss' : loss of the algorithm at each iteration during
            training
        'components' : dimensionality of the representation vectors.
        'max_iter' : maximum number of iterations in training
        """
        return self._metavalues

    @staticmethod
    def _reshape_forward(mat):
        """ transpose a row matrix or tensor to a column matrix """
        if len(mat.shape) == 3:
            return mat.reshape(mat.shape[0], -1).T
        if len(mat.shape) == 2:
            return mat.T
        raise ValueError(f'expected a 2 or 3 dimensional matrix. Got a matrix '
                         'of shape {mat.shape}')

    def _reshape_backward(self, mat):
        """transpose a column matrix to a row matrix / tensor (dependant on
        input to this class on training)"""
        assert len(mat.shape) == 2, 'needs a matrix not a tensor'
        newshape = self._metavalues['image_shape']
        return mat.T.reshape(mat.shape[1], *newshape)
    
    def __str__(self):
        D = self._metavalues
        return f"NMF({D['components']})"



def l2_norm(arr):
    return np.linalg.norm(arr)
