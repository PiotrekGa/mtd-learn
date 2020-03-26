import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
import logging

logger = logging.getLogger(__name__)


def parse_markov_matrix(matrix):

    if not isinstance(matrix, np.ndarray):
        raise TypeError('The matrix should be a numpy array')

    return matrix.reshape(-1, 1).ravel()


class PathEncoder(TransformerMixin, BaseEstimator):

    def __init__(self, lpad_string='Q', sep='>'):
        self.lpad_string = lpad_string
        self.sep = sep
        self.label_dict = None

    def fit(self, x):
        pass

