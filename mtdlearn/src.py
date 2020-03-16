import numpy as np
from itertools import product


class MTD:

    def __init__(self,
                 n_dimensions,
                 order,
                 init_method='random',
                 max_iter=100,
                 min_gain=0.1,
                 verbose=1):

        self.n_dimensions = n_dimensions
        self.order = order
        self.n_parameters = self.order * self.n_dimensions * (self.n_dimensions - 1) + self.order - 1
        self.init_method = init_method
        self.transition_matrix_ = None
        self.n_ = None
        self.p_ = None
        self.log_likelihood = None
        self.aic = None
        self.p_expectation_ = None
        self.p_expectation_direct_ = None
        self.n_direct_ = None
        self.max_iter = max_iter
        self.min_gain = min_gain
        self.verbose = verbose

        if init_method == 'flat':
            self.lambdas_ = np.ones(order) / order
            self.tmatrices_ = np.ones((order, n_dimensions, n_dimensions)) / n_dimensions
        elif init_method == 'random':
            self.lambdas_ = np.random.rand(order)
            self.lambdas_ = self.lambdas_ / self.lambdas_.sum()
            self.tmatrices_ = np.random.rand(order, n_dimensions, n_dimensions)
            self.tmatrices_ = self.tmatrices_ / self.tmatrices_.sum(2).reshape(order, n_dimensions, 1)

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

        self.lambdas_, self.tmatrices_, self.log_likelihood = MTD.fit_one(x,
                                                                          self.indexes,
                                                                          self.order,
                                                                          self.n_dimensions,
                                                                          self.min_gain,
                                                                          self.max_iter,
                                                                          self.tmatrices_,
                                                                          self.lambdas_,
                                                                          self.verbose)

    @staticmethod
    def fit_one(x, indexes, order, n_dimensions, min_gain, max_iter, tmatrices_, lambdas_, verbose):

        if len(x) != len(indexes):
            raise ValueError('input data has wrong length')

        n_direct_ = np.zeros((order, n_dimensions, n_dimensions))
        for i, idx in enumerate(indexes):
            for j, k in enumerate(idx[:-1]):
                n_direct_[j, k, idx[-1]] += x[i]

        iteration = 0
        gain = min_gain * 2
        log_likelihood = MTD._calculate_log_likelihood(indexes,
                                                       x,
                                                       tmatrices_,
                                                       lambdas_)
        while iteration < max_iter and gain > min_gain:
            old_ll = log_likelihood
            p_expectation_, p_expectation_direct_ = MTD._expectation_step(n_dimensions,
                                                                          order,
                                                                          indexes,
                                                                          tmatrices_,
                                                                          lambdas_)
            lambdas_, tmatrices_ = MTD._maximization_step(n_dimensions,
                                                          order,
                                                          indexes,
                                                          x,
                                                          n_direct_,
                                                          p_expectation_,
                                                          p_expectation_direct_,
                                                          tmatrices_,
                                                          lambdas_)
            log_likelihood = MTD._calculate_log_likelihood(indexes,
                                                           x,
                                                           tmatrices_,
                                                           lambdas_)
            gain = log_likelihood - old_ll
            iteration += 1
            if verbose > 0:
                print('iteration:', iteration, '  gain:', round(gain,5), '  ll_value:', round(log_likelihood, 5))

        if iteration == max_iter:
            print('\nWARNING: The model has not converged. Consider increasing the max_iter parameter. \n')

        return lambdas_, tmatrices_, log_likelihood

    @staticmethod
    def _calculate_log_likelihood(indexes,
                                  n_,
                                  tmatrices_,
                                  lambdas_):

        log_likelihood = 0

        for i, idx in enumerate(indexes):
            mtd_value = sum([lam * tmatrices_[i, idx[i], idx[-1]] for i, lam in enumerate(lambdas_)])
            log_likelihood += n_[i] * np.log(mtd_value)

        return log_likelihood

    @staticmethod
    def _expectation_step(n_dimensions,
                          order,
                          indexes,
                          tmatrices_,
                          lambdas_):

        p_expectation_ = np.zeros((n_dimensions ** (order + 1), order))

        for i, idx in enumerate(indexes):
            p_expectation_[i, :] = [lam * tmatrices_[i, idx[i], idx[-1]]
                                    for i, lam
                                    in enumerate(lambdas_)]

        p_expectation_ = p_expectation_ / p_expectation_.sum(axis=1).reshape(-1, 1)

        p_expectation_direct_ = np.zeros((order, n_dimensions, n_dimensions))

        for i, idx in enumerate(indexes):
            for j, k in enumerate(idx[:-1]):
                p_expectation_direct_[j, k, idx[-1]] += p_expectation_[i, j]

        p_expectation_direct_ = p_expectation_direct_ / p_expectation_direct_.sum(axis=0)

        return p_expectation_, p_expectation_direct_

    @staticmethod
    def _maximization_step(n_dimensions,
                           order,
                           indexes,
                           n_,
                           n_direct_,
                           p_expectation_,
                           p_expectation_direct_,
                           tmatrices_,
                           lambdas_):

        denominator = 1 / sum(n_)
        for i, _ in enumerate(lambdas_):
            sum_part = sum([n_[j] * p_expectation_[j, i] for j, _ in enumerate(p_expectation_)])
            lambdas_[i] = denominator * sum_part

        for i, idx in enumerate(indexes):
            for j, k in enumerate(idx[:-1]):
                tmatrices_[j, k, idx[-1]] = n_direct_[j, k, idx[-1]] * p_expectation_direct_[j, k, idx[-1]]

        tmatrices_ = tmatrices_ / tmatrices_.sum(2).reshape(order, n_dimensions, 1)

        return lambdas_, tmatrices_

    def _calculate_aic(self):

        self.aic = -2 * self.log_likelihood + 2 * self.n_parameters