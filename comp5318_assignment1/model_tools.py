import random
import warnings
import numpy as np
from typing import List, Tuple
import h5py
import operator as op
import seaborn as sns
import traceback
from matplotlib import pyplot as plt


class Pipeline:

    def __init__(self, transformations):
        self._transformations = transformations

    def fit(self, X, Y, verbose=False):
        """ fit transformations to data and train the model with the output """
        for name, model in self._transformations[:-1]:
            if verbose:
                print(f'fitting transformation {name} (={model})')
            model.fit(X)
            X = model.transform(X)
        if verbose:
            print(f'fitting estimator {name} (={model})')

        self._transformations[-1][1].fit(X, Y)

    def run_transform(self, data, verbose=False):
        for name, model in self._transformations[:-1]:
            if verbose:
                print(f'transforming with {name} (={model})')
            data = model.transform(data)
        return data

    def predict(self, X, verbose=False, **pred_kwargs):
        return self._transformations[-1][1].predict(
                self.run_transform(X, verbose=verbose), **pred_kwargs)

    def __str__(self):
        return f'Pipeline({", ".join(map(op.itemgetter(0), self._transformations))})'


class CrossValidateClassification:
    """CrossValidateClassification.

    run cross-validation on a dataset with multiple models
    """

    def __init__(self, data: np.ndarray, labels: np.ndarray, n: int = 10, verbose: bool = False):
        """__init__.

        if len(data) > len(labels) data is cut short to only include the first len(labels) examples.

        :param data: input data (all data) MxN
        :type data: np.ndarray
        :param labels: input labels (all labels) N
        :type labels: np.ndarray
        :param n:
        :type n: int
        :return: [(true labels, predicted labels), ...]
        """

        idxs = list(range(len(labels)))
        random.shuffle(idxs)
        size = len(idxs)//n + 1
        validations = [set(idxs[i:i+size]) for i in range(0, len(idxs), size)]
        idxs = set(idxs)
        self.validation_groups = tuple(np.array(list(x)) for x in validations)
        self.data = data
        self.labels = labels
        self.verbose = bool(verbose)

    def run_validation(self, model: object) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """run_validation.

        :param model_class: called to create the model. Must define fit and predict methods.
        :type model_class: object
        :param args: arguments passed to model_class on instanciation.
        :param kwargs: keyword arguments passed to model_class on instanciation.
        :return: list of tuples of predicted observations and true observations
        :rtype: List[Tuple[np.ndarray, np.ndarray]]
        """
        res_true = []
        res_pred = []
        idxs = set(list(range(len(self.labels))))
        for i, idx_test in enumerate(self.validation_groups):
            if self.verbose:
                print(f'running fold #{i+1}     ')

            idx_train    = np.array(list(idxs - set(idx_test)))

            train_data   = self.data[idx_train]
            train_labels = self.labels[idx_train]

            test_data    = self.data[idx_test]
            test_labels  = self.labels[idx_test]

            model.fit(train_data, train_labels)
            pred = model.predict(test_data)
            res_true.append(test_labels)
            res_pred.append(pred)
        return res_true, res_pred

    @staticmethod
    def metrics(true, predicted=None, names=('accuracy',)):
        """metrics.

        :param true:
        :param predicted:
        """

        if predicted is None:  # attempt to unpack
            true, predicted = true

        cm = confusion_matrix(true, predicted)
        acc = np.mean(true == predicted)
        res = []
        if 'accuracy' in names:
            res.append(np.mean(true == predicted))
        return acc, cm  # prec, recall

    @staticmethod
    def aggregate_metrics(true, predicted):
        return np.array([CrossValidateClassification.metrics(t, p) for t, p in zip(true, predicted)])

    def _random_idxs(self, n):
        idxs = list(range(len(self.labels)))
        random.shuffle(idxs)
        return idxs[:n]

    def random_sample(self, n):
        idxs = self._random_idxs(n)
        return self.data[idxs], self.labels[idxs]


def confusion_matrix(true, pred):
    rv = np.zeros([len(true)]*2)
    for t, p in zip(true, pred):
        rv[t, p] += 1
    return rv


def plot_confusion(confusion_matrix, title=None, labels=None, cmap='YlGnBu'):
    ax = sns.heatmap(confusion_matrix, linewidth=0.2, annot=True, cmap=cmap, square=True)
    if labels is not None:
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels, rotation=0)
    if title is not None:
        ax.set_title(title)
    return ax

        
class ModelRunner:

    def __init__(self, *models):
        self.models = models

    def load_data(self):
        """ loads the datasets provided for this assignment """
        if hasattr(self, 'xtr'):
            return  # has already been run

        with h5py.File('./data/Input/train/images_training.h5','r') as H:
            self.xtr = np.copy(H['datatrain'])

        with h5py.File('./data/Input/train/labels_training.h5','r') as H:
            self.ytr = np.copy(H['labeltrain'])

        with h5py.File('./data/Input/test/labels_testing_2000.h5', 'r') as H:
            self.yte = np.copy(H['labeltest'])

        with h5py.File('./data/Input/test/images_testing.h5', 'r') as H:
            self.xte = np.copy(H['datatest'])[:len(self.yte)]

    def run_cv(self, folds=10, verbose=False):
        """ runs n-fold cross validation on the model """
        self.load_data()
        validator = CrossValidateClassification(self.xtr, self.ytr, n=folds, verbose=verbose)
        results = {}
        for model in self.models:
            if verbose:
                print(f'running {model}')
            true, pred = validator.run_validation(model)
            results[str(model)] = CrossValidateClassification.aggregate_metrics(true, pred)
        return results

    def run(self, n=None, verbose=False):
        """run all models and evaluate performance

        :param n: subset of test samples to evaluate model on.
        :param verbose: if true, print progress.
        """
        self.load_data()
        results = {}
        try:
            if n is None:
                n = len(self.xte)
        except ValueError:
            raise ValueError('could not interperate input')
        for model in self.models:
            if verbose:
                print(f'running {model}')
            try:
                model.fit(self.xtr, self.ytr)
            except Exception:
                tb = traceback.format_exc()
                warnings.warn(f'{tb}\nmodel {model} exited unexpectedly.'
                        '\nskipping...')
                continue  # skip this model
            results[str(model)] = CrossValidateClassification.metrics(
                    self.yte[:n], model.predict(self.xte[:n])
            )
            if verbose:
                print(f'{model} got {results[str(model)]}')
        return results


def confusion_matrix(true, pred, labels=None):
    if labels is None:
        labels = list(set(true))
    rv = np.zeros([len(labels)]*2)
    for t, p in zip(true, pred):
        rv[labels.index(p), labels.index(t)] += 1
    return rv


def plot_confusion_matrix(mat, labels):
    '''
    :param mat: confusion matrix
    :param labels: ordered labels to show on graph
    '''
    plt.imshow(mat)
    plt.xticks(range(len(labels)), labels, rotation='vertical')
    plt.yticks([-0.5] + list(range(len(labels))) + [len(labels) - .5],
               [''] + list(labels) + [''],
               rotation='horizontal')


