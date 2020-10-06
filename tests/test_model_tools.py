import numpy as np
from comp5310_assignment1.models import TrivialModel, NMF, KNN, GNB
from comp5310_assignment1.model_tools import Pipeline
import pytest


def modeltest(model, trainx, trainy, testx, expectedtesty, *modelargs, **modelkwargs):
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
    gnb = GNB()
    n = 6
    means = (np.eye(3) * 10000)
    sds = np.ones(3) * .1
    nclasses = means.shape[0]
    x = np.concatenate(np.random.normal(means, sds, (n // nclasses, *means.shape)))
    y = np.tile(np.arange(nclasses), n // nclasses)

    modeltest(gnb, x, y, x, y)
