import numpy as np
from numpy import testing
from comp5318_assignment1.models import TrivialModel, KNN, GNB, np_mode, breakup
from comp5318_assignment1.decomposition import PCA, NMF, IdentityTransformation
from comp5318_assignment1.model_tools import Pipeline, ModelRunner
import pytest

show_results = False

np.set_printoptions(precision=3)


def modeltest(model, trainx, trainy, testx, expectedtesty=None):
    model.fit(trainx, trainy)
    if expectedtesty is not None:
        assert list(model.predict(testx)) == list(expectedtesty)
    else:
        return model.predict(testx)


def test_identitymodel():
    modeltest(
        TrivialModel(rv=10),
        np.zeros(20),
        np.zeros(20),
        np.zeros(5),
        np.ones(5) * 10
    )


def test_GNB():
    """test_GNB. Tests GNB with a dataset that should be extremely easy to
    predict with a gaussian NB model.

    There is some (extremely) small likelihood that this fails even though the
    model works correctly
    """
    gnb = GNB()
    n = 10
    nclasses = 2
    nfeatures = 4
    means = (np.tile(10**np.arange(nfeatures), (nclasses, 1))
            * np.arange(1, nclasses+1)[:,None])
    sds = (np.tile(.1**(np.arange(nfeatures)), (nclasses, 1))
            * np.arange(1, nclasses+1)[:, None])

    x = np.concatenate(np.random.normal(means, sds, (n // nclasses, *means.shape)))
    y = np.tile(np.arange(nclasses), n // nclasses)

    gnb.fit(x, y)
    pred = gnb.predict(x)

    print(gnb.means.shape, means.shape)
    print(gnb.sds.shape, sds.shape)

    print('comparing mean estimates')
    print('========================')
    print(gnb.means - means)
    print(f'largest difference = {np.max(np.abs(gnb.means - means)):.3f}')

    print('\ncomparing sd estimates')
    print('======================')
    print(gnb.sds - sds)
    print(f'largest difference = {np.max(np.abs(gnb.sds - sds)):.3f}')

    print('\ncomparing predictions')
    print('=====================')
    print(y - pred)
    print(f'largest difference = {np.max(np.abs(pred - y)):.3f}')


    preds = modeltest(gnb, x, y, x)
    assert np.all(preds == y)


def test_run_data():
    mr = ModelRunner(KNN(8), GNB())
    for m, v in mr.run(n=30).items():
        print(f'{m: >20} : {v[0]}')
    if show_results:
        assert 0, 'failing to show results'


def test_mode():
    x = np.array([[[ 0,  2,  2],
                   [ 3,  4,  5],
                   [ 6,  7,  2],
                   [ 9, 10, 11],
                   [12, 13, 14]],

                  [[15, 16,  2],
                   [18, 19, 20],
                   [21, 22, 23],
                   [ 2, 25, 26],
                   [27, 28, 29]]])
    assert np_mode(x) == 2

    assert np.all(np_mode(x, axis=(1,2)) == np.ones(2) * 2)

    x = np.array([[[ 0,  2,  2],
                   [ 3,  4,  5],
                   [ 6,  7,  2],
                   [ 9, 10, 11],
                   [12, 13, 14]],

                  [[ 0,  2,  2],
                   [ 3,  4,  5],
                   [ 6,  7,  2],
                   [ 9, 10, 11],
                   [12, 13, 14]],

                  [[15, 16,  2],
                   [18, 19, 20],
                   [21, 22, 23],
                   [ 2, 25, 26],
                   [27, 28, 29]]])

    assert np.all(np_mode(x, axis=0) == np.array(
                  [[ 0,  2,  2],
                   [ 3,  4,  5],
                   [ 6,  7,  2],
                   [ 9, 10, 11],
                   [12, 13, 14]]))

    x = np.random.randint(0, 500, (50, 51, 52, 53))
    assert np_mode(x, axis=0).shape == np.mean(x, axis=0).shape


def test_breakup():
    x = list(range(30))
    exp = list(map(list, (
        range( 0,  4),
        range( 4,  8),
        range( 8, 12),
        range(12, 16),
        range(16, 20),
        range(20, 24),
        range(24, 28),
        range(28, 30)
    )))

    for i, j in zip(breakup(x, 4), exp):
        assert i == j
            

def test_NMF():
    # check that the reduction runs and produces the correct output dimension
    try:
        nmf = NMF(n_components = 5)
    except:
        nmf = NMF(components = 5)
    nmf.fit(np.random.rand(20, 35))
    assert nmf.transform(np.random.rand(200, 35)).shape == (200, 5)


def test_pca():
    try:
        pca = PCA(n_components = 5)
    except:
        pca = PCA(components = 5)
    pca.fit(np.random.rand(20, 35))
    assert pca.transform(np.random.rand(200, 35)).shape == (200, 5)

def test_pipeline():
    def tuplestr(f):
        return str(f), f

    plne = Pipeline([
        tuplestr(IdentityTransformation()),
        tuplestr(TrivialModel(2))
    ])
    plne.fit(np.random.rand(30, 18), np.random.randint(0, 9, 18))
