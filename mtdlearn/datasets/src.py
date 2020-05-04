import numpy as np
from ..mtd import _ChainBase
from typing import Tuple, Optional, List


class ChainGenerator(_ChainBase):
    """
    Class for generating discrete space datasets


    Parameters
    ----------
    values: list or tuple of strings
        States to be used for generation.

    order: int
        Number of lags of the model.

    sep: string, optional (default='>')
        Separator between states.

    min_len: int, optional (default=None)
        Minimal length of the sequence. If None, then equal to order.

    max_len: int, optional (default=None)
        Maximal length of the sequence. If None, then equal to order.

    transition_matrix: NumPy array, optional (default=None)
        Markov model array, to be used for data generation. If None, then the matrix will be generated.

    lambdas: NumPy array, optional (default=None)
        MTD model weights vector, to be used for data generation. If None, then the vector will be generated.

    transition_matrices: NumPy array, optional (default=None)
        MTD model transition matrices, to be used for data generation. If None, then the matrices will be generated.

    random_state: int, optional (default=None)
        Random state, to be used for NumPy seed.

    Example
    ----------
    from mtdlearn.datasets import ChainGenerator

    cg = ChainGenerator(['A', 'B', 'C'], 4)

    x, y = cg.generate_data(100)

"""

    def __init__(self, values: List[str], order: int, sep: str = '>', min_len: Optional[int] = None,
                 max_len: Optional[int] = None, transition_matrix: Optional[np.ndarray] = None,
                 lambdas: Optional[np.ndarray] = None, transition_matrices: Optional[np.ndarray] = None,
                 random_state: Optional[int] = None) -> None:
        super().__init__(order=order, values=values)
        self._n_dimensions = len(values)
        self.order = order
        self.sep = sep
        if not ((min_len is None and max_len is None) or (min_len is not None and max_len is not None)):
            raise ValueError('if min_len is passed max_len has to be specified and vice versa')
        if min_len is None:
            self.min_len = order
        elif min_len < order:
            raise ValueError('min_len cannot be smaller that order')
        else:
            self.min_len = min_len
        if max_len is None:
            self.max_len = order
        elif max_len < order:
            raise ValueError('max_len cannot be smaller that order')
        else:
            self.max_len = max_len
        if max_len is not None and min_len is not None:
            if max_len < min_len:
                raise ValueError('max_len cannot be smaller that min_len')
        self.transition_matrix = transition_matrix
        self.lambdas = lambdas
        self.transition_matrices = transition_matrices
        self.random_state = random_state
        if self.transition_matrix is None:
            self._generate_mtd_model()
            self._create_markov()
        self._label_dict = {i: j for i, j in enumerate(values)}

    def generate_data(self, samples: int, random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate data

        :param samples: int
                        Number of samples to be generated.
        :param random_state: int
                             Random state for generation
        :return: NumPy array of shape (samples, 1)
        :return: NumPy array of shape (samples,)
        """

        if random_state is not None:
            np.random.seed(random_state)

        x = np.random.choice(list(self._label_dict.keys()), (samples, self.order))
        y = self.predict_random(x)

        if self.max_len > self.order:
            x_to_add = np.random.randint(self.min_len, self.max_len + 1, samples)
            x = [np.hstack([np.random.choice(list(self._label_dict.keys()), i[1] - self.order), i[0]])
                 for i
                 in zip(x.tolist(), list(x_to_add))]

        x = [self.sep.join(list(map(self._label_dict.get, i))) for i in x]

        return np.array(x).reshape(-1, 1), np.array(y)

    def _generate_mtd_model(self) -> None:
        if self.random_state is not None:
            np.random.seed(self.random_state)
        if self.lambdas is None:
            lambdas = np.random.rand(self.order)
            self.lambdas = lambdas / lambdas.sum()
        if self.transition_matrices is None:
            transition_matrices = np.random.rand(self.order, self._n_dimensions, self._n_dimensions)
            self.transition_matrices = transition_matrices / transition_matrices.sum(2).reshape(self.order,
                                                                                                self._n_dimensions, 1)


data_values3_order2_full = dict()
data_values3_order2_full['x'] = np.array([['A>A>A'],
                                          ['A>A>B'],
                                          ['A>A>C'],
                                          ['A>B>A'],
                                          ['A>B>A'],
                                          ['A>B>A'],
                                          ['A>B>B'],
                                          ['A>B>B'],
                                          ['A>B>B'],
                                          ['A>B>C'],
                                          ['A>B>C'],
                                          ['A>B>C'],
                                          ['A>C>A'],
                                          ['A>C>A'],
                                          ['A>C>A'],
                                          ['A>C>B'],
                                          ['A>C>B'],
                                          ['A>C>B'],
                                          ['A>C>C'],
                                          ['A>C>C'],
                                          ['A>C>C'],
                                          ['B>A>A'],
                                          ['B>A>A'],
                                          ['B>A>A'],
                                          ['B>A>B'],
                                          ['B>A>B'],
                                          ['B>A>B'],
                                          ['B>A>C'],
                                          ['B>A>C'],
                                          ['B>A>C'],
                                          ['B>B>A'],
                                          ['B>B>A'],
                                          ['B>B>A'],
                                          ['B>B>B'],
                                          ['B>B>B'],
                                          ['B>B>B'],
                                          ['B>B>C'],
                                          ['B>B>C'],
                                          ['B>B>C'],
                                          ['B>C>A'],
                                          ['B>C>A'],
                                          ['B>C>A'],
                                          ['B>C>B'],
                                          ['B>C>B'],
                                          ['B>C>B'],
                                          ['B>C>C'],
                                          ['B>C>C'],
                                          ['B>C>C'],
                                          ['C>A>A'],
                                          ['C>A>A'],
                                          ['C>A>A'],
                                          ['C>A>B'],
                                          ['C>A>B'],
                                          ['C>A>B'],
                                          ['C>A>C'],
                                          ['C>A>C'],
                                          ['C>A>C'],
                                          ['C>B>A'],
                                          ['C>B>A'],
                                          ['C>B>A'],
                                          ['C>B>B'],
                                          ['C>B>B'],
                                          ['C>B>B'],
                                          ['C>B>C'],
                                          ['C>B>C'],
                                          ['C>B>C'],
                                          ['C>C>A'],
                                          ['C>C>A'],
                                          ['C>C>A'],
                                          ['C>C>B'],
                                          ['C>C>B'],
                                          ['C>C>B'],
                                          ['C>C>C'],
                                          ['C>C>C'],
                                          ['C>C>C']])

data_values3_order2_full['y'] = np.array(['A',
                                          'A',
                                          'A',
                                          'A',
                                          'B',
                                          'C',
                                          'A',
                                          'B',
                                          'C',
                                          'A',
                                          'B',
                                          'C',
                                          'A',
                                          'B',
                                          'C',
                                          'A',
                                          'B',
                                          'C',
                                          'A',
                                          'B',
                                          'C',
                                          'A',
                                          'B',
                                          'C',
                                          'A',
                                          'B',
                                          'C',
                                          'A',
                                          'B',
                                          'C',
                                          'A',
                                          'B',
                                          'C',
                                          'A',
                                          'B',
                                          'C',
                                          'A',
                                          'B',
                                          'C',
                                          'A',
                                          'B',
                                          'C',
                                          'A',
                                          'B',
                                          'C',
                                          'A',
                                          'B',
                                          'C',
                                          'A',
                                          'B',
                                          'C',
                                          'A',
                                          'B',
                                          'C',
                                          'A',
                                          'B',
                                          'C',
                                          'A',
                                          'B',
                                          'C',
                                          'A',
                                          'B',
                                          'C',
                                          'A',
                                          'B',
                                          'C',
                                          'A',
                                          'B',
                                          'C',
                                          'A',
                                          'B',
                                          'C',
                                          'A',
                                          'B',
                                          'C'])

data_values3_order2_full['sample_weight'] = np.array([1000,
                                                      1000,
                                                      1000,
                                                      100,
                                                      100,
                                                      100,
                                                      100,
                                                      100,
                                                      100,
                                                      100,
                                                      100,
                                                      100,
                                                      100,
                                                      100,
                                                      100,
                                                      100,
                                                      100,
                                                      100,
                                                      100,
                                                      100,
                                                      100,
                                                      1000,
                                                      0,
                                                      0,
                                                      1000,
                                                      0,
                                                      0,
                                                      1000,
                                                      0,
                                                      0,
                                                      100,
                                                      100,
                                                      100,
                                                      100,
                                                      100,
                                                      100,
                                                      100,
                                                      100,
                                                      100,
                                                      100,
                                                      100,
                                                      100,
                                                      100,
                                                      100,
                                                      100,
                                                      100,
                                                      100,
                                                      100,
                                                      1000,
                                                      0,
                                                      0,
                                                      1000,
                                                      0,
                                                      0,
                                                      1000,
                                                      0,
                                                      0,
                                                      100,
                                                      100,
                                                      100,
                                                      100,
                                                      100,
                                                      100,
                                                      100,
                                                      100,
                                                      100,
                                                      100,
                                                      100,
                                                      100,
                                                      100,
                                                      100,
                                                      100,
                                                      100,
                                                      100,
                                                      100])
