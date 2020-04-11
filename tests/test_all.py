from mtdlearn.mtd import MTD, _ChainBaseEstimator
from mtdlearn.preprocessing import PathEncoder
from mtdlearn.datasets import data_values3_order2_full as data
from mtdlearn.datasets import generate_data
from .data_for_tests import data_for_tests
import numpy as np

x = data['x']
y = data['y']
sample_weight = data['sample_weight']


def test_dataset():
    assert x.shape[0] == y.shape[0]
    assert x.shape[0] == sample_weight.shape[0]


def test_generate_data1():
    x_gen, y_gen = generate_data(('A', 'B', 'C'), '>', 1, 10, 3, 1000, 0.95)
    assert x_gen.shape[0] == y_gen.shape[0]
    assert y_gen.shape[0] == 1000
    assert max([len(i[0].split('>')) for i in x_gen]) <= 10
    assert min([len(i[0].split('>')) for i in x_gen]) >= 1


def test_generate_data2():
    x_gen, y_gen = generate_data(('A', 'B', 'C', 'D'), '*', 5, 5, 5, 100, 1.0)
    assert x_gen.shape[0] == y_gen.shape[0]
    assert y_gen.shape[0] == 100
    assert max([len(i[0].split('*')) for i in x_gen]) == 5
    assert min([len(i[0].split('*')) for i in x_gen]) == 5
    for i, row in enumerate(x_gen):
        assert row[0].split('*')[0] == y_gen[i]


def test_path_encoder1():
    x_gen, y_gen = generate_data(('A', 'B', 'C'), '*', 1, 3, 3, 100, 0.95)
    pe = PathEncoder(3, '*', 'X')
    pe.fit(x_gen, y_gen)
    x_tr, y_tr = pe.transform(x_gen, y_gen)
    x_gen_rep, y_gen_rep = pe.inverse_transform(x_tr, y_tr)

    assert list(pe.label_dict.keys()) == ['A', 'B', 'C', 'X']
    assert list(pe.label_dict.values()) == [0, 1, 2, 3]
    assert list(pe.label_dict_inverse.values()) == ['A', 'B', 'C', 'X']
    assert list(pe.label_dict_inverse.keys()) == [0, 1, 2, 3]
    assert x_tr.shape[0] == x_gen.shape[0]
    assert y_tr.shape[0] == y_gen.shape[0]
    assert x_tr.shape[1] == 3
    assert list(np.unique(x_tr)) == [0, 1, 2, 3]
    assert list(np.unique(y_tr)) == [0, 1, 2]
    assert x_gen.shape == x_gen_rep.shape
    assert y_gen.shape == y_gen_rep.shape
    assert list(np.unique(y_gen_rep)) == ['A', 'B', 'C']
    assert len(list(set(x_gen_rep[0][0].split('*')) & {'A', 'B', 'C', 'X'})) > 0


def test_path_encoder2():
    x_gen, y_gen = generate_data(('A', 'B', 'C', 'D'), '>', 1, 10, 3, 300, 0.95)
    pe = PathEncoder(5, '>', 'X')
    pe.fit(x_gen, y_gen)
    x_tr, y_tr = pe.transform(x_gen, y_gen)
    x_gen_rep, y_gen_rep = pe.inverse_transform(x_tr, y_tr)

    assert list(pe.label_dict.keys()) == ['A', 'B', 'C', 'D', 'X']
    assert list(pe.label_dict.values()) == [0, 1, 2, 3, 4]
    assert list(pe.label_dict_inverse.values()) == ['A', 'B', 'C', 'D', 'X']
    assert list(pe.label_dict_inverse.keys()) == [0, 1, 2, 3, 4]
    assert x_tr.shape[0] == x_gen.shape[0]
    assert y_tr.shape[0] == y_gen.shape[0]
    assert x_tr.shape[1] == 5
    assert list(np.unique(x_tr)) == [0, 1, 2, 3, 4]
    assert list(np.unique(y_tr)) == [0, 1, 2, 3]
    assert x_gen.shape == x_gen_rep.shape
    assert y_gen.shape == y_gen_rep.shape
    assert list(np.unique(y_gen_rep)) == ['A', 'B', 'C', 'D']
    assert len(list(set(x_gen_rep[0][0].split('>')) & {'A', 'B', 'C', 'D', 'X'})) > 0


def test_chain_aggregator1():
    x_gen = np.array([[0, 0], [0, 1]])
    y_gen = np.array([0, 1])
    ca = _ChainBaseEstimator()
    result = ca.aggregate_chain(x_gen, y_gen)
    assert np.array_equal(result, np.array([1, 0, 0, 1, 0, 0, 0, 0]))


def test_chain_aggregator2():
    x_gen = np.array([[1, 0, 0], [2, 2, 2]])
    y_gen = np.array([1, 2])
    sample_weight_gen = np.array([100, 99])
    ca = _ChainBaseEstimator()
    result = ca.aggregate_chain(x_gen, y_gen, sample_weight_gen)
    assert result[28] == 100
    assert result[80] == 99
    assert result.shape == (81,)
    assert result.sum() == 199


def test_create_indexes():
    mtd = MTD(4, 3)
    assert len(mtd.indexes_) == 256


def test_manual_exp_max():
    indexes = data_for_tests['indexes']
    transition_matrices = data_for_tests['transition_matrices'].copy()
    lambdas = data_for_tests['lambdas'].copy()
    expected_p_array = data_for_tests['expected_p_array']
    expected_p_direct_array = data_for_tests['expected_p_direct_array']
    n_passes = data_for_tests['n_passes']
    n_passes_direct = data_for_tests['n_passes_direct']
    expected_lambdas = data_for_tests['expected_lambdas']
    expected_transition_matrices = data_for_tests['expected_transition_matrices']
    log_likelihood1 = data_for_tests['log_likelihood1']
    log_likelihood2 = data_for_tests['log_likelihood2']

    mtd = MTD(2, 2)

    log_likelihood_start = mtd._calculate_log_likelihood(indexes, n_passes, transition_matrices, lambdas)

    expectation_matrix, expectation_matrix_direct = mtd._expectation_step(2, 2, indexes, transition_matrices, lambdas)

    lambdas_out, transition_matrices_out = mtd._maximization_step(2, 2,
                                                                  indexes,
                                                                  n_passes,
                                                                  n_passes_direct,
                                                                  expectation_matrix,
                                                                  expectation_matrix_direct,
                                                                  transition_matrices,
                                                                  lambdas)

    log_likelihood_end = mtd._calculate_log_likelihood(indexes, n_passes, transition_matrices_out, lambdas_out)

    assert np.isclose((log_likelihood1 - log_likelihood_start), 0)
    assert np.isclose((log_likelihood2 - log_likelihood_end), 0)
    assert np.isclose((expectation_matrix - expected_p_array), np.zeros((8, 2))).min()
    assert np.isclose((expectation_matrix_direct - expected_p_direct_array), np.zeros((2, 2, 2))).min()
    assert np.isclose((expected_lambdas - lambdas_out), np.zeros((2,))).min()
    assert np.isclose((expected_transition_matrices - transition_matrices_out), np.zeros((2, 2, 2))).min()


def test_one_fit():
    indexes = data_for_tests['indexes']
    transition_matrices = data_for_tests['transition_matrices'].copy()
    lambdas = data_for_tests['lambdas'].copy()
    n_passes = data_for_tests['n_passes']
    n_passes_direct = data_for_tests['n_passes_direct']
    expected_lambdas = data_for_tests['expected_lambdas']
    expected_transition_matrices = data_for_tests['expected_transition_matrices']
    expected_log_likelihood = data_for_tests['log_likelihood2']

    mtd = MTD(2, 2)
    log_likelihood, lambdas_out, transition_matrices_out = mtd._fit_one(x=n_passes,
                                                                        indexes=indexes,
                                                                        order=2,
                                                                        n_dimensions=2,
                                                                        min_gain=1.0,
                                                                        max_iter=1,
                                                                        verbose=0,
                                                                        n_direct=n_passes_direct,
                                                                        lambdas=lambdas,
                                                                        transition_matrices=transition_matrices)

    assert np.isclose(log_likelihood - expected_log_likelihood, 0.0)
    assert np.isclose(lambdas_out - expected_lambdas, np.zeros(2)).min()
    assert np.isclose(expected_transition_matrices - transition_matrices_out, np.zeros((2, 2))).min()


def test_fit_with_init():
    transition_matrices = data_for_tests['transition_matrices'].copy()
    lambdas = data_for_tests['lambdas'].copy()
    pe = PathEncoder(2)
    pe.fit(x, y)
    x_tr, y_tr = pe.transform(x, y)
    mtd = MTD(2, 2, max_iter=0, verbose=0, lambdas_init=lambdas, transition_matrices_init=transition_matrices)
    mtd.fit(x_tr, y_tr)
    assert np.isclose(mtd.transition_matrices - transition_matrices, np.zeros((2, 2))).min()
    assert np.isclose(mtd.lambdas - lambdas, np.zeros(2)).min()


def test_ex_max():
    for seed in range(20):
        np.random.seed(seed)
        pe = PathEncoder(2)
        pe.fit(x, y)
        x_tr, y_tr = pe.transform(x, y)
        mtd = MTD(3, 2, verbose=0, number_of_initiations=1)
        mtd.fit(x_tr, y_tr)
        assert mtd.lambdas.shape == (2,)
        assert np.isclose(sum(mtd.lambdas), 1.0)
        assert max(mtd.lambdas) <= 1
        assert min(mtd.lambdas) >= 0
        assert np.isclose(sum(mtd.transition_matrices[0, 0, :]), 1.0)
        assert mtd.transition_matrices.shape == (2, 3, 3)
        assert mtd.transition_matrices.min() >= 0
        assert mtd.transition_matrices.max() <= 1


def test_create_markov():
    pe = PathEncoder(2)
    pe.fit(x, y)
    x_tr, y_tr = pe.transform(x, y)
    mtd = MTD(3, 2, verbose=0)
    mtd.fit(x_tr, y_tr)
    assert mtd.transition_matrix.max() <= 1.0
    assert mtd.transition_matrix.min() >= 0.0
    assert np.isclose(mtd.transition_matrix.sum(1).max(), 1.0)
    assert mtd.transition_matrix.shape == (9, 3)


def test_n_parameters():
    mtd = MTD(4, 1)
    assert mtd.n_parameters_ == 12
    mtd = MTD(4, 2)
    assert mtd.n_parameters_ == 21
    mtd = MTD(4, 3)
    assert mtd.n_parameters_ == 30
    mtd = MTD(4, 4)
    assert mtd.n_parameters_ == 39
    mtd = MTD(4, 5)
    assert mtd.n_parameters_ == 48


def test_predict():
    transition_matrices = data_for_tests['transition_matrices'].copy()
    lambdas = data_for_tests['lambdas'].copy()
    pe = PathEncoder(2)
    pe.fit(x, y)
    x_tr, y_tr = pe.transform(x, y)
    mtd = MTD(2, 2,
              max_iter=0,
              verbose=0,
              number_of_initiations=1,
              lambdas_init=lambdas,
              transition_matrices_init=transition_matrices)
    mtd.fit(x_tr, y_tr)
    assert np.array_equal(mtd.predict(np.array([[0, 0], [0, 1], [1, 0], [1, 1]])), np.array([1, 1, 1, 0]))


def test_predict_proba():
    transition_matrices = data_for_tests['transition_matrices'].copy()
    lambdas = data_for_tests['lambdas'].copy()
    pe = PathEncoder(2)
    pe.fit(x, y)
    x_tr, y_tr = pe.transform(x, y)
    mtd = MTD(2, 2,
              max_iter=0,
              verbose=0,
              number_of_initiations=1,
              lambdas_init=lambdas,
              transition_matrices_init=transition_matrices)
    mtd.fit(x_tr, y_tr)
    assert np.isclose(mtd.predict_proba(np.array([[0, 0], [0, 1], [1, 0], [1, 1]])), np.array([[0.22, 0.78],
                                                                                               [0.46, 0.54],
                                                                                               [0.34, 0.66],
                                                                                               [0.58, 0.42]])).min()