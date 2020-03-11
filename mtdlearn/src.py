import numpy as np
from itertools import product


class MTD:

    def __init__(self,
                 n_dimensions,
                 order,
                 lambdas_init='flat',
                 tmatrices_init='flat'):

        self.n_dimensions = n_dimensions
        self.order = order
        self.lambdas_init = lambdas_init
        self.tmatrices_init = tmatrices_init
        self.transition_matrix_ = None
        self.n_ = None
        self.p_ = None
        if lambdas_init == 'flat':
            self.lambdas_ = np.ones(order) / order
        if tmatrices_init == 'flat':
            self.tmatrices_ = np.ones((order, n_dimensions, n_dimensions)) / n_dimensions

        idx_gen = product(range(self.n_dimensions), repeat=self.order + 1)

        self.indexes = []
        for i in idx_gen:
            self.indexes.append(i)

    def create_markov(self):

        array_coords = product(range(self.n_dimensions), repeat=self.order)

        tmatrix_list = []
        for idx in array_coords:
            tmatrix_list.append(np.dot(np.array([self.tmatrices_[i, idx[i], :] for i in range(self.order)]).T,
                                       self.lambdas_))
        self.transition_matrix_ = np.array(tmatrix_list)

    def fit(self, x):
        if len(x) != len(self.indexes):
            raise ValueError('input data has wrong length')

        self.n_ = x
        self.p_ = x / sum(x)