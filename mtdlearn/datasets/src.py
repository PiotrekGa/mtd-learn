import numpy as np
from ..mtd import _ChainBaseEstimator
from itertools import product


class ChainGenerator(_ChainBaseEstimator):

    def __init__(self, values, sep, order, min_len=None, max_len=None, transition_matrix=None, lambdas=None,
                 transition_matrices=None,
                 random_state=None):
        super().__init__(n_dimensions=len(values), order=order)
        self.values = values
        self.sep = sep
        self.order = order
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

    def generate_data(self, samples, random_state=None):

        if random_state is not None:
            np.random.seed(random_state)

        x = []
        cnt = 0
        while cnt < samples:
            cnt += 1
            seq_list = np.random.choice(list(self._label_dict.keys()), self.order)
            x.append(seq_list)

        x = np.array(x).reshape(-1, self.order)
        y = self._predict_random(x)

        x_to_add = np.random.randint(self.min_len, self.max_len + 1, samples)

        x_to_encode = [np.hstack([np.random.choice(list(self._label_dict.keys()), i[1] - self.order), i[0]])
                       for i
                       in zip(x.tolist(), list(x_to_add))]

        x = [self.sep.join(list(map(self._label_dict.get, i))) for i in x_to_encode]

        return np.array(x).reshape(-1, 1), np.array(y)

    def _predict_random(self, x):
        prob = self.predict_proba(x)
        x_new = [np.random.choice(self.values, p=i) for i in prob]
        return x_new

    def _generate_mtd_model(self):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        if self.lambdas is None:
            lambdas = np.random.rand(self.order)
            self.lambdas = lambdas / lambdas.sum()
        if self.transition_matrices is None:
            transition_matrices = np.random.rand(self.order, self.n_dimensions, self.n_dimensions)
            self.transition_matrices = transition_matrices / transition_matrices.sum(2).reshape(self.order,
                                                                                                self.n_dimensions, 1)

    def _create_markov(self):

        array_coords = product(range(self.n_dimensions), repeat=self.order)

        transition_matrix_list = []
        for idx in array_coords:
            t_matrix_part = np.array([self.transition_matrices[i, idx[i], :] for i in range(self.order)]).T
            transition_matrix_list.append(np.dot(t_matrix_part,
                                                 self.lambdas))
        self.transition_matrix = np.array(transition_matrix_list)


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
