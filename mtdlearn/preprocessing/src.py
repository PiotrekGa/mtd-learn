import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator


class PathEncoder(TransformerMixin, BaseEstimator):

    """
    Encoding of string paths into NumPy array

    Parameters
    ----------
    order: int
        Number of states to be encoded
    sep: str, optional (default='>')
        Separator between states
    r_just_string: instr, optional (default='>')t
        String to be placed where path is too short

    Attributes
    ----------
    label_dict: dict
        Dictionary mapping original values to encoded integers

    label_dict_inverse: dict
        Dictionary mapping encoded integers to original values

    Example
    ----------
    import numpy as np
    from mtdlearn.preprocessing import PathEncoder

    x = np.array([['A>B'], ['B>A'], ['C']])
    y = np.array(['A', 'B', 'D'])

    pe = PathEncoder(2)
    pe.fit(x, y)
    x_tr, y_tr = pe.transform(x, y)

    """

    def __init__(self, order, sep='>', r_just_string='null'):
        self.order = order
        self.sep = sep
        self.r_just_string = r_just_string
        self.label_dict = None
        self.label_dict_inverse = None

    def fit(self, x, y=None):
        """
        Fit encoder.

        :param x: NumPy array of shape (n_samples, 1)
                  Array with string paths
        :param y: NumPy array of shape (n_samples,), default=None
                  Target values
        :return: self
        """
        x = np.char.split(x, self.sep)

        unique_keys = [[self.r_just_string]]
        for i in x:
            unique_keys.append(i[0][-self.order:])

        unique_keys = list(set(sum(unique_keys, [])))
        if y is not None:
            y = np.unique(y).tolist()
            unique_keys.extend(y)
            unique_keys = list(set(unique_keys))

        unique_keys.sort()
        self.label_dict = {k: i for i, k in enumerate(unique_keys)}
        self.label_dict_inverse = {i: k for i, k in enumerate(unique_keys)}

        return self

    def transform(self, x, y=None):
        """
        Transform inputs from string paths into integer array.

        :param x: NumPy array of shape (n_samples, 1)
                  Array with string paths
        :param y: NumPy array of shape (n_samples,), default=None
                  Target values
        :return: NumPy array of shape (n_samples, order)
        :return: NumPy array of shape (n_samples,), if y is not None
        """

        x_new = []
        for i in x[:, 0]:
            values_list = list(map(self.label_dict.get, i.split(self.sep)[-self.order:]))
            while len(values_list) < self.order:
                values_list = [self.label_dict[self.r_just_string]] + values_list
            x_new.append(values_list)

        if y is None:
            return np.array(x_new)
        else:
            return np.array(x_new), np.vectorize(self.label_dict.get)(y)

    def inverse_transform(self, x, y=None):
        """
        Transform inputs from integer array into string paths.

        :param x: NumPy array of shape (n_samples, order)
                  Array with integer encoded paths
        :param y: NumPy array of shape (n_samples,), default=None
                  Encoded target values
        :return: NumPy array of shape (n_samples, 1)
        :return: NumPy array of shape (n_samples,), if y is not None
        """
        x_rev = []
        for i in x.tolist():
            seq_mapped = self.sep.join(list(map(self.label_dict_inverse.get, i)))
            x_rev.append(seq_mapped)

        if y is None:
            return np.array(x_rev).reshape(-1, 1)
        else:
            return np.array(x_rev).reshape(-1, 1), np.vectorize(self.label_dict_inverse.get)(y)
