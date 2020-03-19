import numpy as np
from itertools import product
from joblib import Parallel, delayed


class MTD:

    def __init__(self,
                 n_dimensions,
                 order,
                 init_method='random',
                 number_of_initiations=10,
                 max_iter=100,
                 min_gain=0.1,
                 verbose=1,
                 n_jobs=-1):

        self.n_dimensions = n_dimensions
        self.order = order
        self.n_parameters_ = self.order * self.n_dimensions * (self.n_dimensions - 1) + self.order - 1
        self.init_method = init_method
        self.number_of_initiations = number_of_initiations
        self.transition_matrix = None
        self.log_likelihood = None
        self.aic = None
        self.lambdas = None
        self.transition_matrices = None
        self.max_iter = max_iter
        self.min_gain = min_gain
        self.verbose = verbose
        self.n_jobs = n_jobs

        idx_gen = product(range(self.n_dimensions), repeat=self.order + 1)

        self.indexes_ = []
        for i in idx_gen:
            self.indexes_.append(i)

        if init_method not in ['random', 'flat']:
            raise ValueError('no such initialization method')

    def fit(self, x):

        if len(x) != len(self.indexes_):
            raise ValueError('input data has wrong length')

        n_direct = np.zeros((self.order, self.n_dimensions, self.n_dimensions))
        for i, idx in enumerate(self.indexes_):
            for j, k in enumerate(idx[:-1]):
                n_direct[j, k, idx[-1]] += x[i]

        candidates = Parallel(n_jobs=self.n_jobs)(delayed(MTD.fit_one)(x,
                                                                       self.indexes_,
                                                                       self.order,
                                                                       self.n_dimensions,
                                                                       self.min_gain,
                                                                       self.max_iter,
                                                                       self.verbose,
                                                                       self.init_method,
                                                                       n_direct)
                                                  for _ in range(self.number_of_initiations))

        self.log_likelihood, self.lambdas, self.transition_matrices = self._select_the_best_candidate(candidates)

        if self.verbose > 0:
            print('best value:', self.log_likelihood)

        self._create_markov()
        self._calculate_aic()

    def predict_proba(self, x):

        x = ''.join(x.astype(str))
        idx = int(x, 2)

        return self.transition_matrix[idx, :]

    def predict(self, x):

        prob = self.predict_proba(x)

        return prob.argmax()

    @staticmethod
    def fit_one(x, indexes, order, n_dimensions, min_gain, max_iter, verbose, init_method, n_direct):

        if init_method == 'flat':
            lambdas = np.ones(order) / order
            transition_matrices = np.ones((order, n_dimensions, n_dimensions)) / n_dimensions
        else:
            lambdas = np.random.rand(order)
            lambdas = lambdas / lambdas.sum()
            transition_matrices = np.random.rand(order, n_dimensions, n_dimensions)
            transition_matrices = transition_matrices / transition_matrices.sum(2).reshape(order, n_dimensions, 1)

        iteration = 0
        gain = min_gain * 2
        log_likelihood = MTD._calculate_log_likelihood(indexes,
                                                       x,
                                                       transition_matrices,
                                                       lambdas)
        while iteration < max_iter and gain > min_gain:
            old_ll = log_likelihood
            p_expectation, p_expectation_direct = MTD._expectation_step(n_dimensions,
                                                                        order,
                                                                        indexes,
                                                                        transition_matrices,
                                                                        lambdas)
            lambdas, transition_matrices = MTD._maximization_step(n_dimensions,
                                                                  order,
                                                                  indexes,
                                                                  x,
                                                                  n_direct,
                                                                  p_expectation,
                                                                  p_expectation_direct,
                                                                  transition_matrices,
                                                                  lambdas)
            log_likelihood = MTD._calculate_log_likelihood(indexes,
                                                           x,
                                                           transition_matrices,
                                                           lambdas)
            gain = log_likelihood - old_ll
            iteration += 1

            if verbose > 1:
                print('iteration:', iteration, '  gain:', round(gain, 5), '  ll_value:', round(log_likelihood, 5))

        if iteration == max_iter:
            print('\nWARNING: The model has not converged. Consider increasing the max_iter parameter.\n')

        if verbose > 0:
            print("log-likelihood value:", log_likelihood)

        return log_likelihood, lambdas, transition_matrices

    @staticmethod
    def _calculate_log_likelihood(indexes,
                                  n_occurrence,
                                  transition_matrices,
                                  lambdas):

        log_likelihood = 0

        for i, idx in enumerate(indexes):
            mtd_value = sum([lam * transition_matrices[i, idx[i], idx[-1]] for i, lam in enumerate(lambdas)])
            log_likelihood += n_occurrence[i] * np.log(mtd_value)

        return log_likelihood

    @staticmethod
    def _expectation_step(n_dimensions,
                          order,
                          indexes,
                          transition_matrices,
                          lambdas):

        p_expectation = np.zeros((n_dimensions ** (order + 1), order))

        for i, idx in enumerate(indexes):
            p_expectation[i, :] = [lam * transition_matrices[i, idx[i], idx[-1]]
                                   for i, lam
                                   in enumerate(lambdas)]

        p_expectation = p_expectation / p_expectation.sum(axis=1).reshape(-1, 1)

        p_expectation_direct = np.zeros((order, n_dimensions, n_dimensions))

        for i, idx in enumerate(indexes):
            for j, k in enumerate(idx[:-1]):
                p_expectation_direct[j, k, idx[-1]] += p_expectation[i, j]

        p_expectation_direct = p_expectation_direct / p_expectation_direct.sum(axis=0)

        return p_expectation, p_expectation_direct

    @staticmethod
    def _maximization_step(n_dimensions,
                           order,
                           indexes,
                           n_occurrence,
                           n_direct,
                           p_expectation,
                           p_expectation_direct,
                           transition_matrices,
                           lambdas):

        denominator = 1 / sum(n_occurrence)
        for i, _ in enumerate(lambdas):
            sum_part = sum([n_occurrence[j] * p_expectation[j, i] for j, _ in enumerate(p_expectation)])
            lambdas[i] = denominator * sum_part

        for i, idx in enumerate(indexes):
            for j, k in enumerate(idx[:-1]):
                transition_matrices[j, k, idx[-1]] = n_direct[j, k, idx[-1]] * p_expectation_direct[j, k, idx[-1]]

        transition_matrices = transition_matrices / transition_matrices.sum(2).reshape(order, n_dimensions, 1)

        return lambdas, transition_matrices

    def _create_markov(self):

        array_coords = product(range(self.n_dimensions), repeat=self.order)

        transition_matrix_list = []
        for idx in array_coords:
            t_matrix_part = np.array([self.transition_matrices[i, idx[i], :] for i in range(self.order)]).T
            transition_matrix_list.append(np.dot(t_matrix_part,
                                                 self.lambdas))
        self.transition_matrix = np.array(transition_matrix_list)

    def _calculate_aic(self):

        self.aic = -2 * self.log_likelihood + 2 * self.n_parameters_

    @staticmethod
    def _select_the_best_candidate(candidates):

        log_likelihood = candidates[0][0]
        lambdas = candidates[0][1]
        transition_matrices = candidates[0][2]

        for c in candidates[1:]:
            if c[0] > log_likelihood:
                log_likelihood = c[0]
                lambdas = c[1]
                transition_matrices = c[2]

        return log_likelihood, lambdas, transition_matrices

