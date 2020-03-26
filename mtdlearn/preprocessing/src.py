import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
import logging

logger = logging.getLogger(__name__)


def parse_markov_matrix(matrix):

    if not isinstance(matrix, np.ndarray):
        raise TypeError('The matrix should be a numpy array')

    return matrix.reshape(-1, 1).ravel()


class PathEncoder(TransformerMixin, BaseEstimator):

    def __init__(self, order, sep='>', r_just_string='null'):
        self.order = order
        self.sep = sep
        self.r_just_string = r_just_string
        self.label_dict = None

    def fit(self, x):

        x = np.char.split(x, self.sep)
        unique_keys = set(sum(x[:, 0], []))
        self.label_dict = {k: i for i, k in enumerate(unique_keys)}
        self.label_dict[self.r_just_string] = max(self.label_dict.values()) + 1

        return x
