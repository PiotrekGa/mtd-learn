import numpy as np
from itertools import product
from joblib import Parallel, delayed


class MTD:

    def __init__(self,
                 n_dimensions,
                 order,
                 init_method='random',
                 init_num=10,
                 max_iter=100,
                 min_gain=0.1,
                 verbose=1,
                 n_jobs=-1):

        self.n_dimensions = n_dimensions
        self.order = order
        self.n_parameters = self.order * self.n_dimensions * (self.n_dimensions - 1) + self.order - 1
        self.init_method = init_method
        self.init_num = init_num
        self.transition_matrix_ = None
        self.log_likelihood = None
        self.aic = None
        self.lambdas_ = None
        self.transition_matrices_ = None
        self.max_iter = max_iter
        self.min_gain = min_gain
        self.verbose = verbose
        self.n_jobs = n_jobs

        idx_gen = product(range(self.n_dimensions), repeat=self.order + 1)

        self.indexes = []
        for i in idx_gen:
            self.indexes.append(i)

    def create_markov(self):

        array_coords = product(range(self.n_dimensions), repeat=self.order)

        transition_matrix_list = []
        for idx in array_coords:
            t_matrix_part = np.array([self.transition_matrices_[i, idx[i], :] for i in range(self.order)]).T
            transition_matrix_list.append(np.dot(t_matrix_part,
                                                 self.lambdas_))
        self.transition_matrix_ = np.array(transition_matrix_list)

    def fit(self, x):

        if len(x) != len(self.indexes):
            raise ValueError('input data has wrong length')

        n_direct_ = np.zeros((self.order, self.n_dimensions, self.n_dimensions))
        for i, idx in enumerate(self.indexes):
            for j, k in enumerate(idx[:-1]):
                n_direct_[j, k, idx[-1]] += x[i]

        candidates = Parallel(n_jobs=self.n_jobs)(delayed(MTD.fit_one)(x,
                                                                       self.indexes,
                                                                       self.order,
                                                                       self.n_dimensions,
                                                                       self.min_gain,
                                                                       self.max_iter,
                                                                       self.verbose,
                                                                       self.init_method,
                                                                       n_direct_) for _ in range(self.init_num))

        self.log_likelihood = candidates[0][0]
        self.lambdas_ = candidates[0][1]
        self.transition_matrices_ = candidates[0][2]

        for c in candidates[1:]:
            if c[0] > self.log_likelihood:
                self.log_likelihood = c[0]
                self.lambdas_ = c[1]
                self.transition_matrices_ = c[2]

        if self.verbose > 0:
            print('best value:', self.log_likelihood)

    @staticmethod
    def fit_one(x, indexes, order, n_dimensions, min_gain, max_iter, verbose, init_method, n_direct_):

        if init_method == 'flat':
            lambdas_ = np.ones(order) / order
            transition_matrices_ = np.ones((order, n_dimensions, n_dimensions)) / n_dimensions
        elif init_method == 'random':
            lambdas_ = np.random.rand(order)
            lambdas_ = lambdas_ / lambdas_.sum()
            transition_matrices_ = np.random.rand(order, n_dimensions, n_dimensions)
            transition_matrices_ = transition_matrices_ / transition_matrices_.sum(2).reshape(order, n_dimensions, 1)
        else:
            raise ValueError('no such initialization method')

        iteration = 0
        gain = min_gain * 2
        log_likelihood = MTD._calculate_log_likelihood(indexes,
                                                       x,
                                                       transition_matrices_,
                                                       lambdas_)
        while iteration < max_iter and gain > min_gain:
            old_ll = log_likelihood
            p_expectation_, p_expectation_direct_ = MTD._expectation_step(n_dimensions,
                                                                          order,
                                                                          indexes,
                                                                          transition_matrices_,
                                                                          lambdas_)
            lambdas_, transition_matrices_ = MTD._maximization_step(n_dimensions,
                                                                    order,
                                                                    indexes,
                                                                    x,
                                                                    n_direct_,
                                                                    p_expectation_,
                                                                    p_expectation_direct_,
                                                                    transition_matrices_,
                                                                    lambdas_)
            log_likelihood = MTD._calculate_log_likelihood(indexes,
                                                           x,
                                                           transition_matrices_,
                                                           lambdas_)
            gain = log_likelihood - old_ll
            iteration += 1

            if verbose > 1:
                print('iteration:', iteration, '  gain:', round(gain, 5), '  ll_value:', round(log_likelihood, 5))

        if iteration == max_iter:
            print('\nWARNING: The model has not converged. Consider increasing the max_iter parameter.\n')

        if verbose > 0:
            print("log-likelihood value:", log_likelihood)

        return log_likelihood, lambdas_, transition_matrices_

    @staticmethod
    def _calculate_log_likelihood(indexes,
                                  n_,
                                  transition_matrices_,
                                  lambdas_):

        log_likelihood = 0

        for i, idx in enumerate(indexes):
            mtd_value = sum([lam * transition_matrices_[i, idx[i], idx[-1]] for i, lam in enumerate(lambdas_)])
            log_likelihood += n_[i] * np.log(mtd_value)

        return log_likelihood

    @staticmethod
    def _expectation_step(n_dimensions,
                          order,
                          indexes,
                          transition_matrices_,
                          lambdas_):

        p_expectation_ = np.zeros((n_dimensions ** (order + 1), order))

        for i, idx in enumerate(indexes):
            p_expectation_[i, :] = [lam * transition_matrices_[i, idx[i], idx[-1]]
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
                           transition_matrices_,
                           lambdas_):

        denominator = 1 / sum(n_)
        for i, _ in enumerate(lambdas_):
            sum_part = sum([n_[j] * p_expectation_[j, i] for j, _ in enumerate(p_expectation_)])
            lambdas_[i] = denominator * sum_part

        for i, idx in enumerate(indexes):
            for j, k in enumerate(idx[:-1]):
                transition_matrices_[j, k, idx[-1]] = n_direct_[j, k, idx[-1]] * p_expectation_direct_[j, k, idx[-1]]

        transition_matrices_ = transition_matrices_ / transition_matrices_.sum(2).reshape(order, n_dimensions, 1)

        return lambdas_, transition_matrices_

    def _calculate_aic(self):

        self.aic = -2 * self.log_likelihood + 2 * self.n_parameters
