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

        result = []
        unique_keys = []
        for i in x:
            sequence = i[0]
            if len(sequence) < self.order:
                temp = [self.r_just_string for i in range(self.order - len(i[0]))]
                temp.extend(i[0])
            else:
                temp = sequence
            result.append(self.sep.join(temp))
            unique_keys.append(temp)

        result.sort()

        unique_keys = set(sum(unique_keys, []))
        self.label_dict = {k: i for i, k in enumerate(unique_keys)}

        return result
