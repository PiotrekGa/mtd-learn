import numpy as np
from itertools import product
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from datetime import datetime
from typing import List, Tuple, Optional

np.seterr(divide='ignore', invalid='ignore')


class _ChainBase(BaseEstimator):
    """
    Base class for chain processing and estimation.


    Parameters
    ----------

    order: int
        Number of lags of the model.
    """

    def __init__(self, order: int = None, values: List = None) -> None:
        self.log_likelihood = None
        self.aic = None
        self.bic = None
        self._n_parameters = None
        self.samples = None
        self._n_dimensions = None
        self.order = order
        self._transition_matrix = None
        self.transition_matrices = None
        self.lambdas = None
        self._indexes = None
        self.values = values
        self.expanded_matrix = None

    def _create_indexes(self) -> None:
        idx_gen = product(range(self._n_dimensions), repeat=self.order + 1)
        self._indexes = [i for i in idx_gen]

    @property
    def transition_matrix(self) -> np.ndarray:
        return self._transition_matrix

    @transition_matrix.setter
    def transition_matrix(self, new_transition_matrix: np.ndarray) -> None:
        self._transition_matrix = new_transition_matrix

    def _calculate_dimensions(self, array: np.ndarray) -> None:
        max_value = array.max()
        min_value = array.min()
        n_dim = np.unique(array).shape[0]
        if min_value != 0:
            raise ValueError('Lowest label should be equal to zero')
        if max_value + 1 != n_dim:
            raise ValueError('Highest label should be equal to number of unique labels minus one')
        self._n_dimensions = n_dim

    def _aggregate_chain(self,
                         x: np.ndarray,
                         sample_weight: Optional[np.ndarray] = None) -> np.ndarray:

        if sample_weight is None:
            sample_weight = np.ones(x.shape[0], dtype=np.int)

        n_columns = x.shape[1]
        values_dict = {i: 0 for i in range(self._n_dimensions ** (x.shape[1]))}
        idx = [self._n_dimensions ** i for i in range(n_columns)]

        idx = np.array(idx[::-1])
        data_indexes = np.dot(x, idx)

        for n, index in enumerate(data_indexes):
            values_dict[index] += sample_weight[n]

        return np.array(list(values_dict.values()))

    def _calculate_aic(self) -> None:

        self.aic = -2 * self.log_likelihood + 2 * self._n_parameters

    def _calculate_bic(self) -> None:

        self.bic = -2 * self.log_likelihood + np.log(self.samples) * self._n_parameters

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """
        Probability estimates.

        The returned estimates for all states are ordered by the
        label of state.
        :param x: NumPy array of shape (n_samples, order)
        :return:  NumPy array of shape (n_samples, _n_dimensions)
        """

        if self.order == 0:
            x = np.zeros((x.shape[0], 1), dtype=int)

        idx = [self._n_dimensions ** i for i in range(x.shape[1])]

        idx = np.array(idx[::-1])
        indexes = np.dot(x, idx)

        return self.transition_matrix[indexes, :]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict state.

        :param x: NumPy array of shape (n_samples, order)
        :return:  NumPy array of shape (n_samples,)
        """

        prob = self.predict_proba(x)

        return prob.argmax(axis=1)

    def _calculate_log_likelihood(self, transition_matrix_num: np.ndarray) -> None:
        logs = np.nan_to_num(np.log(self.transition_matrix), nan=0.0)
        self.log_likelihood = (transition_matrix_num * logs).sum()

    def _create_transition_matrix(self,
                                  x: np.ndarray,
                                  y: np.ndarray,
                                  sample_weight: np.ndarray) -> np.ndarray:
        x = np.hstack([x, y.reshape(-1, 1)])
        self._calculate_dimensions(x)
        transition_matrix = self._aggregate_chain(x, sample_weight)
        transition_matrix_num = transition_matrix.reshape(-1, self._n_dimensions)
        self.transition_matrix = transition_matrix_num / transition_matrix_num.sum(1).reshape(-1, 1)
        return transition_matrix_num

    def _check_and_reshape_input(self, x: np.ndarray) -> np.ndarray:
        if x.shape[1] > self.order:
            print(f'WARNING: The input has too many columns. Expected: {self.order}, got: {x.shape[1]}. '
                  f'The columns were trimmed.')
            x = x[:, -self.order:]
        if x.shape[1] < self.order:
            raise ValueError(f'WARNING: The input has less columns than order. Expected: {self.order}, '
                             f'got: {x.shape[1]}.')
        return x

    def _create_markov(self) -> None:

        array_coords = product(range(self._n_dimensions), repeat=self.order)

        transition_matrix_list = []
        for idx in array_coords:
            t_matrix_part = np.array([self.transition_matrices[i, idx[i], :] for i in range(self.order)]).T
            transition_matrix_list.append(np.dot(t_matrix_part,
                                                 self.lambdas))
        self.transition_matrix = np.array(transition_matrix_list)

    def predict_random(self, x: np.ndarray) -> List:
        """
        Return state sampled from probability distribution from transition matrix. Used primarily for data generation.

        :param x: NumPy array of shape (n_samples, order)
        :return:  NumPy array of shape (n_samples,)
        """
        prob = self.predict_proba(x)
        x_new = [np.random.choice(self.values, p=i) for i in prob]
        return x_new

    def create_expanded_matrix(self) -> None:
        """
        Transforms transition matrix into first order transition matrix.
        See 1.1 in The Mixture Transition Distribution Model for High-Order Markov Chains and Non-Gaussian Time Series.

        :return: self
        """

        if self.order > 1:
            idx_gen = product(range(self._n_dimensions), repeat=self.order)
            idx = [i for i in idx_gen]

            self.expanded_matrix = np.zeros((len(idx), len(idx)))
            for i, row in enumerate(idx):
                for j, col in enumerate(idx):
                    if row[-(self.order - 1):] == col[:(self.order - 1)]:
                        self.expanded_matrix[i, j] = self.transition_matrix[i, j % self._n_dimensions]

        else:
            self.expanded_matrix = self.transition_matrix.copy()


class MTD(_ChainBase):
    """
    Mixture Transition Distribution (MTD) model with separate transition matrices for each lag.


    Parameters
    ----------

    order: int
        Number of lags of the model.

    number_of_initiations: int, optional (default=10)
        Number of parameters sets to be initiated. 1 is minimum.

    max_iter: int, optional (default=100)
        Maximum number of iterations for the EM algorithm.

    min_gain: float, optional (default=0.1)
        Minimum change of the log-likelihood function value for a step in the EM optimization algorithm.

    lambdas_init: NumPy array, optional (default=None)
        Starting value for lambdas.

    transition_matrices_init: NumPy array, optional (default=None)
        Starting value for transition_matrices.

    verbose: int, optional (default=1)
        Controls the verbosity when fitting and predicting.

    n_jobs: int, optional (default=-1)
        Number of threads to be used for estimation. Every initiation set can be estimated on one thread only.


    Attributes
    ----------

    _n_dimensions: int
        Number of states of the process.

    _n_parameters: int
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

    bic: float
        Value of the Bayesian Information Criterion (BIC)

    Example
    ----------
    import numpy as np
    from mtdlearn.mtd import MTD

    np.random.seed(42)

    _n_dimensions = 3
    order = 2

    m = MTD(_n_dimensions, order, n_jobs=-1)

    x = np.array([[0, 0],
                  [1, 1],
                  [2, 2],
                  [0, 1],
                  [2, 1],
                  [2, 0],
                  [0, 1],
                  [2, 1],
                  [1, 1],
                  [1, 0]])
    y = np.array([0, 0, 2, 1, 1, 2, 0, 1, 2, 1])

    m.fit(x, y)

    x = np.array([[0, 0],
                  [1, 1],
                  [2, 2]])

    m.predict_proba(x)

    m.predict(x)

    References
    ----------
    -- [1] S. Lèbre, P Bourguinon "An EM algorithm for estimation in the Mixture Transition Distribution model", 2008

    """

    def __init__(self,
                 order: int,
                 number_of_initiations: int = 10,
                 max_iter: int = 100,
                 min_gain: float = 0.1,
                 lambdas_init: Optional[np.ndarray] = None,
                 transition_matrices_init: Optional[np.ndarray] = None,
                 verbose: np.int = 1,
                 n_jobs: int = -1) -> None:

        super().__init__(order)
        self.number_of_initiations = number_of_initiations
        self.lambdas_init = lambdas_init
        self.transition_matrices_init = transition_matrices_init
        self.max_iter = max_iter
        self.min_gain = min_gain
        self.verbose = verbose
        self.n_jobs = n_jobs

    def fit(self,
            x: np.ndarray,
            y: np.ndarray,
            sample_weight: Optional[np.ndarray] = None) -> None:
        """
        Fit MTD model.

        Note that the labels (for combined x and y) has to start from zero and be consecutive.

        :param x: NumPy array of shape (n_samples, order)
                  Training data
        :param y: NumPy array of shape (n_samples,)
                  Target values
        :param sample_weight: NumPy array of shape (n_samples,), default=None
                              Individual weights for each sample
        :return: self
        """

        if sample_weight is not None:
            self.samples = sample_weight.sum()
        else:
            self.samples = y.shape[0]

        x = self._check_and_reshape_input(x)
        x = np.hstack([x, y.reshape(-1, 1)])
        self._calculate_dimensions(x)
        x = self._aggregate_chain(x, sample_weight)

        self._create_indexes()

        n_direct = [x.reshape(-1,
                              self._n_dimensions,
                              self._n_dimensions ** (self.order - i - 1),
                              self._n_dimensions).sum(0).sum(1) for i in range(self.order)]
        n_direct = np.array(n_direct)

        candidates = Parallel(n_jobs=self.n_jobs)(delayed(MTD._fit_one)(x,
                                                                        self._indexes,
                                                                        self.order,
                                                                        self._n_dimensions,
                                                                        self.min_gain,
                                                                        self.max_iter,
                                                                        self.verbose,
                                                                        n_direct,
                                                                        self.lambdas_init,
                                                                        self.transition_matrices_init)
                                                  for _ in range(self.number_of_initiations))

        self.log_likelihood, self.lambdas, self.transition_matrices = self._select_the_best_candidate(candidates)

        if self.verbose > 0:
            print(f'log-likelihood value: {self.log_likelihood}')

        self._create_markov()
        self._n_parameters = (1 + self.order * (self._n_dimensions - 1)) * (self._n_dimensions - 1)
        self._calculate_aic()
        self._calculate_bic()

    @staticmethod
    def _fit_one(x: np.ndarray,
                 indexes: List[Tuple[int]],
                 order: int,
                 n_dimensions: int,
                 min_gain: float,
                 max_iter: int,
                 verbose: int,
                 n_direct: np.ndarray,
                 lambdas: Optional[np.ndarray] = None,
                 transition_matrices: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray, np.ndarray]:

        if lambdas is None:
            lambdas = np.random.rand(order)
            lambdas = lambdas / lambdas.sum()
        if transition_matrices is None:
            transition_matrices = np.random.rand(order, n_dimensions, n_dimensions)
            transition_matrices = transition_matrices / transition_matrices.sum(2).reshape(order, n_dimensions, 1)

        iteration = 0
        gain = min_gain * 2
        log_likelihood = MTD._calculate_log_likelihood_mtd(indexes,
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
                                                                  x,
                                                                  n_direct,
                                                                  p_expectation,
                                                                  p_expectation_direct)
            log_likelihood = MTD._calculate_log_likelihood_mtd(indexes,
                                                               x,
                                                               transition_matrices,
                                                               lambdas)
            gain = log_likelihood - old_ll
            iteration += 1

            if verbose > 1:
                current_time = datetime.now()
                print(f'{current_time.time()} iteration: {iteration}  '
                      f'gain: {round(gain, 5)} ll_value: {round(log_likelihood, 5)}')

        if iteration == max_iter:
            print('WARNING: The model has not converged. Consider increasing the max_iter parameter.')

        if verbose > 0:
            print(f"log-likelihood value: {log_likelihood}")

        return log_likelihood, lambdas, transition_matrices

    @staticmethod
    def _calculate_log_likelihood_mtd(indexes: List[Tuple[int]],
                                      n_occurrence: np.ndarray,
                                      transition_matrices: np.ndarray,
                                      lambdas: np.ndarray) -> float:

        log_likelihood = 0

        for i, idx in enumerate(indexes):
            if n_occurrence[i] > 0:
                mtd_value = sum([lam * transition_matrices[i, idx[i], idx[-1]] for i, lam in enumerate(lambdas)])
                log_likelihood += n_occurrence[i] * np.log(mtd_value)

        return log_likelihood

    @staticmethod
    def _expectation_step(n_dimensions: int,
                          order: int,
                          indexes: List[Tuple[int]],
                          transition_matrices: np.ndarray,
                          lambdas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        p_expectation = []
        for i in range(n_dimensions):
            parts = product(*transition_matrices[:, :, i].tolist())
            p_expectation.append(np.array([i for i in parts]))

        p_expectation = np.hstack(p_expectation).reshape(-1, order) * lambdas

        p_expectation = p_expectation / p_expectation.sum(axis=1).reshape(-1, 1)
        p_expectation = np.nan_to_num(p_expectation, nan=1. / order)

        p_expectation_direct = [p_expectation[:, i].reshape(-1,
                                                            n_dimensions,
                                                            n_dimensions ** (order - i - 1),
                                                            n_dimensions).sum(0).sum(1) for i in range(order)]

        p_expectation_direct = np.array(p_expectation_direct)
        p_expectation_direct = p_expectation_direct / p_expectation_direct.sum(axis=0)

        return p_expectation, p_expectation_direct

    @staticmethod
    def _maximization_step(n_dimensions: int,
                           order: int,
                           n_occurrence: np.ndarray,
                           n_direct: np.ndarray,
                           p_expectation: np.ndarray,
                           p_expectation_direct: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        denominator = 1 / sum(n_occurrence)
        sum_part = (p_expectation * n_occurrence.reshape(-1, 1)).sum(0)
        lambdas = denominator * sum_part

        transition_matrices = n_direct * p_expectation_direct
        transition_matrices = transition_matrices / transition_matrices.sum(2).reshape(order, n_dimensions, 1)

        return lambdas, transition_matrices

    @staticmethod
    def _select_the_best_candidate(candidates: List) -> Tuple[float, np.ndarray, np.ndarray]:

        log_likelihood = candidates[0][0]
        lambdas = candidates[0][1]
        transition_matrices = candidates[0][2]

        for c in candidates[1:]:
            if c[0] > log_likelihood:
                log_likelihood = c[0]
                lambdas = c[1]
                transition_matrices = c[2]

        return log_likelihood, lambdas, transition_matrices


class MarkovChain(_ChainBase):
    """
    Markov Chain model.


    Parameters
    ----------

    order: int
        Number of lags of the model.

    verbose: int, optional (default=1)
        Controls the verbosity when fitting and predicting.


    Attributes
    ----------

    _n_dimensions: int
        Number of states of the process.

    _n_parameters: int
        Number of independent parameters of the model

    transition_matrix: NumPy array
        Markov Chain transition matrix

    log_likelihood: float
        Log-likelihood the the Markov Chain model

    aic: float
        Value of the Akaike's Information Criterion (AIC)

    bic: float
        Value of the Bayesian Information Criterion (BIC)
    """

    def __init__(self, order: int, verbose: int = 1) -> None:

        super().__init__(order)
        self.verbose = verbose

    def fit(self,
            x: np.ndarray,
            y: np.ndarray,
            sample_weight: Optional[np.ndarray] = None) -> None:
        """
        Fit Markov Chain model.

        :param x: NumPy array of shape (n_samples, order)
                  Training data
        :param y: NumPy array of shape (n_samples,)
                  Target values
        :param sample_weight: NumPy array of shape (n_samples,), default=None
                              Individual weights for each sample
        :return: self
        """

        if sample_weight is not None:
            self.samples = sample_weight.sum()
        else:
            self.samples = y.shape[0]

        x = self._check_and_reshape_input(x)
        self._n_dimensions = np.unique(np.hstack([x, y.reshape(-1, 1)])).shape[0]
        self._n_parameters = (self._n_dimensions ** self.order) * (self._n_dimensions - 1)
        self._create_indexes()

        transition_matrix_num = self._create_transition_matrix(x, y, sample_weight)

        self._calculate_log_likelihood(transition_matrix_num)
        self._calculate_aic()
        self._calculate_bic()

        if self.verbose > 0:
            print(f'log-likelihood value: {self.log_likelihood}')


class RandomWalk(_ChainBase):
    """
    Random Walk model.


    Parameters
    ----------

    verbose: int, optional (default=1)
        Controls the verbosity when fitting and predicting.


    Attributes
    ----------

    _n_dimensions: int
        Number of states of the process.

    _n_parameters: int
        Number of independent parameters of the model

    transition_matrix: NumPy array
        Random Walk probability matrix

    log_likelihood: float
        Log-likelihood the the Random Walk model

    aic: float
        Value of the Akaike's Information Criterion (AIC)

    bic: float
        Value of the Bayesian Information Criterion (BIC)
    """

    def __init__(self,
                 verbose: int = 1) -> None:

        super().__init__(0)
        self.verbose = verbose

    def fit(self,
            y: np.ndarray,
            sample_weight: Optional[np.ndarray] = None) -> None:
        """
        Fit Random Walk model.

        :param y: NumPy array of shape (n_samples,)
                  Target values
        :param sample_weight: NumPy array of shape (n_samples,), default=None
                              Individual weights for each sample
        :return: self
        """
        if sample_weight is not None:
            self.samples = sample_weight.sum()
        else:
            self.samples = y.shape[0]

        self._n_dimensions = np.unique(y).shape[0]
        self._n_parameters = self._n_dimensions - 1
        self._create_indexes()

        x = np.array([[] for _ in range(len(y))])

        transition_matrix_num = self._create_transition_matrix(x, y, sample_weight)

        self._calculate_log_likelihood(transition_matrix_num)
        self._calculate_aic()
        self._calculate_bic()

        if self.verbose > 0:
            print(f'log-likelihood value: {self.log_likelihood}')
