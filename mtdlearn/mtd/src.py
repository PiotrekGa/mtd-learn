import numpy as np
from itertools import product
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator

np.seterr(divide='ignore', invalid='ignore')


class ChainBaseEstimator(BaseEstimator):

    def __init__(self):
        self.log_likelihood = None
        self.aic = None
        self.bic = None
        self.n_parameters_ = None
        self.samples = None

    @staticmethod
    def aggregate_chain(x, y, sample_weight=None):

        if sample_weight is None:
            sample_weight = np.ones(y.shape[0], dtype=np.int)

        matrix = np.hstack([x, y.reshape(-1, 1)])
        n_unique = len(np.unique(matrix))
        n_columns = matrix.shape[1]
        values_dict = {i: 0 for i in range(n_unique ** (x.shape[1] + 1))}

        idx = []
        for i in range(n_columns):
            idx.append(n_unique ** i)
        idx = np.array(idx[::-1])
        indexes = np.dot(matrix, idx)

        for n, index in enumerate(indexes):
            values_dict[index] += sample_weight[n]

        return np.array(list(values_dict.values()))

    def _calculate_aic(self):

        self.aic = -2 * self.log_likelihood + 2 * self.n_parameters_

    def _calculate_bic(self):

        self.bic = -2 * self.log_likelihood + np.log(self.samples) * self.n_parameters_


class MTD(ChainBaseEstimator):
    """
    Mixture Transition Distribution (MTD) model with separate transition matrices for each lag.


    Parameters
    ----------
    n_dimensions: int
        Number of states of the process.

    order: int
        Number of lags of the model.

    number_of_initiations: int, optional (default=10)
        Number of parameters sets to be initiated. 1 is minimum.

    max_iter: int, optional (default=100)
        Maximum number of iterations for the EM algorithm.

    min_gain: float, optional (default=0.1)
        Minimum change of the log-likelihood function value for a step in the EM optimization algorithm.

    verbose: int, optional (default=1)
        Controls the verbosity when fitting and predicting.

    n_jobs: int, optional (default=-1)
        Number of threads to be used for estimation. Every initiation set can be estimated on one thread only.


    Attributes
    ----------
    n_parameters_: int
        Number of independent parameters of the model following [1] section 2

    lambdas: NumPy array
        Weights vector

    transition_matrices: NumPy array
        Transition matrices of the model

    transition_matrix: NumPy array
        Reconstructed Markov Chain transition matrix

    log_likelihood: float
        Log-likelihood the the MTD model

    aic: float
        Value of the Akaike's Information Criterion (AIC)

    Example
    ----------
    >>> import numpy as np
    >>> from mtdlearn.mtd import MTD

    >>> np.random.seed(42)

    >>> n_dimensions = 3
    >>> order = 2

    >>> m = MTD(n_dimensions, order, n_jobs=-1)

    >>> x = np.array([[0, 0],
    ...               [1, 1],
    ...               [2, 2],
    ...               [0, 1],
    ...               [2, 1],
    ...               [2, 0],
    ...               [0, 1],
    ...               [2, 1],
    ...               [1, 1],
    ...               [1, 0]])
    >>> y = np.array([0, 0, 2, 1, 1, 2, 0, 1, 2, 1])

    >>> m.fit(x, y)

    >>> x = np.array([[0, 0],
    ...               [1, 1],
    ...               [2, 2]])

    >>> m.predict_proba(x)

    >>> m.predict(x)

    References
    ----------
    -- [1] S. Lèbre, P Bourguinon "An EM algorithm for estimation in the Mixture Transition Distribution model", 2008

    """

    def __init__(self, n_dimensions, order, number_of_initiations=10, max_iter=100, min_gain=0.1, lambdas_init=None,
                 transition_matrices_init=None, verbose=1, n_jobs=-1):

        super().__init__()
        self.n_dimensions = n_dimensions
        self.order = order
        self.n_parameters_ = (1 + self.order * (self.n_dimensions - 1)) * (self.n_dimensions - 1)
        self.number_of_initiations = number_of_initiations
        self.transition_matrix = None
        self.lambdas = None
        self.transition_matrices = None
        self.lambdas_init = lambdas_init
        self.transition_matrices_init = transition_matrices_init
        self.max_iter = max_iter
        self.min_gain = min_gain
        self.verbose = verbose
        self.n_jobs = n_jobs

        idx_gen = product(range(self.n_dimensions), repeat=self.order + 1)

        self.indexes_ = []
        for i in idx_gen:
            self.indexes_.append(i)

    def fit(self, x, y, sample_weight=None):

        if sample_weight is not None:
            self.samples = sample_weight.sum()
        else:
            self.samples = y.shape[0]

        x = self.aggregate_chain(x, y, sample_weight)

        n_direct = np.zeros((self.order, self.n_dimensions, self.n_dimensions))
        for i, idx in enumerate(self.indexes_):
            for j, k in enumerate(idx[:-1]):
                n_direct[j, k, idx[-1]] += x[i]

        candidates = Parallel(n_jobs=self.n_jobs)(delayed(MTD._fit_one)(x,
                                                                        self.indexes_,
                                                                        self.order,
                                                                        self.n_dimensions,
                                                                        self.min_gain,
                                                                        self.max_iter,
                                                                        self.verbose,
                                                                        n_direct,
                                                                        self.lambdas_init,
                                                                        self.transition_matrices_init)
                                                  for _ in range(self.number_of_initiations))

        self.log_likelihood, self.lambdas, self.transition_matrices = self._select_the_best_candidate(candidates)

        if self.verbose > 0:
            print(f'best value: {self.log_likelihood}')

        self._create_markov()
        self._calculate_aic()
        self._calculate_bic()

    def predict_proba(self, x):

        idx = []
        for i in range(x.shape[1]):
            idx.append(self.n_dimensions ** i)
        idx = np.array(idx[::-1])
        indexes = np.dot(x, idx)

        return self.transition_matrix[indexes, :]

    def predict(self, x):

        prob = self.predict_proba(x)

        return prob.argmax(axis=1)

    @staticmethod
    def _fit_one(x, indexes, order, n_dimensions, min_gain, max_iter, verbose, n_direct, lambdas=None,
                 transition_matrices=None):

        if lambdas is None:
            lambdas = np.random.rand(order)
            lambdas = lambdas / lambdas.sum()
        if transition_matrices is None:
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
                print(f'iteration: {iteration}  gain: {round(gain, 5)} ll_value: {round(log_likelihood, 5)}')

        if iteration == max_iter:
            print('WARNING: The model has not converged. Consider increasing the max_iter parameter.')

        if verbose > 0:
            print(f"log-likelihood value: {log_likelihood}")

        return log_likelihood, lambdas, transition_matrices

    @staticmethod
    def _calculate_log_likelihood(indexes,
                                  n_occurrence,
                                  transition_matrices,
                                  lambdas):

        log_likelihood = 0

        for i, idx in enumerate(indexes):
            if n_occurrence[i] > 0:
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
        p_expectation = np.nan_to_num(p_expectation, nan=1. / order)

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
