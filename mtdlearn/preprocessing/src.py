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

    def fit(self, x, y=None):

        x = np.char.split(x, self.sep)

        unique_keys = [[self.r_just_string]]
        for i in x:
            unique_keys.append(i[0])

        unique_keys = list(set(sum(unique_keys, [])))
        if y is not None:
            y = np.unique(y).tolist()
            unique_keys.extend(y)
            unique_keys = list(set(unique_keys))

        unique_keys.sort()
        self.label_dict = {k: i for i, k in enumerate(unique_keys)}

        return self

    def transform(self, x, y=None):

        x_new = []

        for i in x[:, 0]:
            values_list = list(map(self.label_dict.get, i.split(self.sep)))
            while len(values_list) < self.order:
                values_list = [self.label_dict[self.r_just_string]] + values_list
            x_new.append(values_list)

        if y is None:
            return np.array(x_new)
        else:
            return np.array(x_new), np.vectorize(self.label_dict.get)(y)
