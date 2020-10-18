from sklearn.decomposition import NMF
from sklearn.decomposition import PCA

class IdentityTransformation:
    def __init__(self):
        pass
    def fit(X, y=None, **params):
        print(params)
        pass
    def transform(X, y=None):
        return X
    def inverse_transform(W):
        return W

    def __str__(self):
        return 'IdentityTransformation()'


class PCA:

    def __init__(self, components=None, normalize=True):
        self.k = components
        self.normalize = normalize

    def fit(X, y=None):
        pass

    def transform(X):
        pass

    def inverse_transform(W):
        pass

