import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
import logging

logger = logging.getLogger(__name__)


def parse_markov_matrix(matrix):

    if not isinstance(matrix, np.ndarray):
        raise TypeError('The matrix should be a numpy array')

    return matrix.reshape(-1, 1).ravel()


class PathEncoder(TransformerMixin, BaseEstimator):

    def __init__(self, order, r_just_string='Q', sep='>'):
        self.order = order
        self.r_just_string = r_just_string
        self.sep = sep
        self.label_dict = None

    def fit(self, x):
        x = self._r_just_with_sep(x)

    def _r_just_with_sep(self, x):
        x = np.char.rjust(x, (2 * self.order - 1), self.r_just_string)
        x = np.char.replace(x, self.r_just_string + self.r_just_string, self.r_just_string + self.sep)
        return x
