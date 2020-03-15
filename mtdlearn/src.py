import numpy as np
from itertools import product


class MTD:

    def __init__(self,
                 n_dimensions,
                 order,
                 lambdas_init='flat',
                 tmatrices_init='flat',
                 max_iter=100,
                 min_gain=0.1):

        self.n_dimensions = n_dimensions
        self.order = order
        self.n_parameters = self.order * self.n_dimensions * (self.n_dimensions - 1) + self.order - 1
        self.lambdas_init = lambdas_init
        self.tmatrices_init = tmatrices_init
        self.transition_matrix_ = None
        self.n_ = None
        self.p_ = None
        self.log_likelihood = None
        self.aic = None
        self.p_expectation_ = None
        self.p_expectation_direct_ = None
        self.p_expectation_direct_tot_ = None
        self.max_iter = max_iter
        self.min_gain = min_gain

        if lambdas_init == 'flat':
            self.lambdas_ = np.ones(order) / order
        if tmatrices_init == 'flat':
            self.tmatrices_ = np.ones((order, n_dimensions, n_dimensions)) / n_dimensions

        idx_gen = product(range(self.n_dimensions), repeat=self.order + 1)

        self.indexes = []
        for i in idx_gen:
            self.indexes.append(i)

        self.n_direct_ = None
        self.n_direct_tot_ = None

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

        self.n_direct_ = np.zeros((self.order, self.n_dimensions, self.n_dimensions))
        for i, idx in enumerate(self.indexes):
            for j, k in enumerate(idx[:-1]):
                self.n_direct_[j, k, idx[-1]] += self.n_[i]

        self.n_direct_tot_ = self.n_direct_.sum(axis=2)

        iteration = 0
        gain = self.min_gain * 2
        self._calculate_log_likelihood()
        while iteration < self.max_iter and gain > self.min_gain:
            old_ll = self.log_likelihood
            self._expectation_step()
            self._maximization_step()
            self._calculate_log_likelihood()
            gain = self.log_likelihood - old_ll
            iteration += 1

    def _calculate_log_likelihood(self):

        self.log_likelihood = 0

        for i, idx in enumerate(self.indexes):
            mtd_value = sum([lam * self.tmatrices_[i, idx[i], idx[-1]] for i, lam in enumerate(self.lambdas_)])
            self.log_likelihood += self.n_[i] * np.log(mtd_value)

    def _calculate_aic(self):

        self.aic = -2 * self.log_likelihood + 2 * self.n_parameters

    def _expectation_step(self):

        self.p_expectation_ = np.zeros((self.n_dimensions ** (self.order + 1), self.order))

        for i, idx in enumerate(self.indexes):
            self.p_expectation_[i, :] = [lam * self.tmatrices_[i, idx[i], idx[-1]]
                                         for i, lam
                                         in enumerate(self.lambdas_)]

        self.p_expectation_ = self.p_expectation_ / self.p_expectation_.sum(axis=1).reshape(-1, 1)

        self.p_expectation_direct_ = np.zeros((self.order, self.n_dimensions, self.n_dimensions))

        for i, idx in enumerate(self.indexes):
            for j, k in enumerate(idx[:-1]):
                self.p_expectation_direct_[j, k, idx[-1]] += self.p_expectation_[i, j]

        self.p_expectation_direct_ = self.p_expectation_direct_ / \
                                     self.p_expectation_direct_.sum(axis=0)

        self.p_expectation_direct_tot_ = np.zeros((self.order, self.n_dimensions))

        for i, idx in enumerate(self.indexes):
            for j, k in enumerate(idx[:-1]):
                self.p_expectation_direct_tot_[j, k] += self.p_expectation_[i, j]

        self.p_expectation_direct_tot_ = self.p_expectation_direct_tot_ / self.p_expectation_direct_tot_.sum(axis=0)

    def _maximization_step(self):

        denominator = 1 / sum(self.n_)
        for i, _ in enumerate(self.lambdas_):
            sum_part = sum([self.n_[j] * self.p_expectation_[j, i] for j, _ in enumerate(self.p_expectation_)])
            self.lambdas_[i] = denominator * sum_part

        for i, idx in enumerate(self.indexes):
            for j, k in enumerate(idx[:-1]):
                numerator = self.n_direct_[j, k, idx[-1]] * self.p_expectation_direct_[j, k, idx[-1]]
                denominator = self.n_direct_tot_[j, k] * self.p_expectation_direct_tot_[j, k]
                self.tmatrices_[j, k, idx[-1]] = numerator / denominator
