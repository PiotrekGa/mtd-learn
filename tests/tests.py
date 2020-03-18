from mtdlearn import MTD
import pytest
import numpy as np


def test_create_indexes():
    mtd = MTD(4, 3)
    assert len(mtd.indexes_) == 256


def test_input_len_error():
    
    with pytest.raises(ValueError):
        mtd = MTD(4, 3, verbose=0)
        mtd.fit(np.array([1, 2, 3]))


def test_input_len_no_error():
    mtd = MTD(3, 2, verbose=0)
    mtd.fit(np.array([i for i in range(27)]))


def test_ex_max():

    for seed in range(100):
        np.random.seed(seed)
        mtd = MTD(3, 2, verbose=0, number_of_initiations=1)
        mtd.fit(np.random.randint(0, 100, 27, ))
        assert mtd.lambdas.shape == (2,)
        assert np.isclose(sum(mtd.lambdas), 1.0)
        assert max(mtd.lambdas) <= 1
        assert min(mtd.lambdas) >= 0
        assert np.isclose(sum(mtd.transition_matrices[0, 0, :]), 1.0)
        assert mtd.transition_matrices.shape == (2, 3, 3)
        assert mtd.transition_matrices.min() >= 0
        assert mtd.transition_matrices.max() <= 1


def test_create_markov():
    mtd = MTD(3, 2, verbose=0)
    mtd.fit(np.array([i for i in range(27)]))
    mtd.create_markov()
    assert mtd.transition_matrix.max() <= 1.0
    assert mtd.transition_matrix.min() >= 0.0
    assert np.isclose(mtd.transition_matrix.sum(1).max(), 1.0)
    assert mtd.transition_matrix.shape == (9, 3)


def test_aic():
    mtd1 = MTD(3, 2, verbose=0)
    mtd1.fit(np.array([i for i in range(27)]))
    mtd1._calculate_aic()
    mtd2 = MTD(3, 3, verbose=0)
    mtd2.fit(np.array([i for i in range(81)]))
    mtd2._calculate_aic()
    assert mtd1.aic < mtd2.aic


def test_criterion():
    n_dimensions = 2
    order = 3
    mtd1 = MTD(n_dimensions, order, verbose=0)
    x = np.array([[100, 900],
                  [100, 900],
                  [900, 100],
                  [900, 100],
                  [100, 900],
                  [100, 900],
                  [900, 100],
                  [900, 100]]).reshape(-1, 1).ravel()  # this is generated by MTD model with lambdas = [0, 1, 0]
    mtd1.fit(x)
    mtd1._calculate_aic()

    order = 2
    mtd2 = MTD(n_dimensions, order, verbose=0)
    x = x.reshape(2, -1).sum(0)
    mtd2.fit(x)
    mtd2._calculate_aic()

    order = 1
    mtd3 = MTD(n_dimensions, order, verbose=0)
    x = x.reshape(2, -1).sum(0)
    mtd3.fit(x)
    mtd3._calculate_aic()

    assert mtd3.aic > mtd1.aic > mtd2.aic


def test_one_fit_random():

    n_dimensions = 2
    order = 3
    mtd = MTD(n_dimensions, order, verbose=0)
    x = np.array([[100, 900],
                  [100, 900],
                  [900, 100],
                  [900, 100],
                  [100, 900],
                  [100, 900],
                  [900, 100],
                  [900, 100]]).reshape(-1, 1).ravel()

    n_direct = np.array([[[2000., 2000.],
                          [2000., 2000.]],
                          [[400., 3600.],
                          [3600.,  400.]],
                          [[2000., 2000.],
                          [2000., 2000.]]])

    log_likelihood, lambdas, transition_matrices = mtd.fit_one(x,
                                                               mtd.indexes_,
                                                               order,
                                                               n_dimensions,
                                                               0.1,
                                                               100,
                                                               0,
                                                               'random',
                                                               n_direct)

    assert lambdas[0] < lambdas[1]
    assert lambdas[2] < lambdas[1]


def test_one_fit_flat():

    n_dimensions = 2
    order = 3
    mtd = MTD(n_dimensions, order, verbose=0)
    x = np.array([[100, 900],
                  [100, 900],
                  [900, 100],
                  [900, 100],
                  [100, 900],
                  [100, 900],
                  [900, 100],
                  [900, 100]]).reshape(-1, 1).ravel()

    n_direct = np.array([[[2000., 2000.],
                          [2000., 2000.]],
                          [[400., 3600.],
                          [3600.,  400.]],
                          [[2000., 2000.],
                          [2000., 2000.]]])

    log_likelihood, lambdas, transition_matrices = mtd.fit_one(x,
                                                               mtd.indexes_,
                                                               order,
                                                               n_dimensions,
                                                               0.1,
                                                               100,
                                                               0,
                                                               'flat',
                                                               n_direct)

    assert lambdas[0] < lambdas[1]
    assert lambdas[2] < lambdas[1]


def test_init_method_error():

    with pytest.raises(ValueError):
        mtd = MTD(4, 3, init_method='a')


def test_final_estimates():

    np.random.seed(42)
    n_dimensions = 2
    order = 3

    mtd = MTD(n_dimensions, order, n_jobs=1, max_iter=1000, verbose=0)

    x = np.array([[200, 800],
                  [200, 800],
                  [900, 100],
                  [900, 100],
                  [200, 800],
                  [200, 800],
                  [900, 100],
                  [900, 100]]).reshape(-1, 1).ravel() * 100

    mtd.fit(x)

    assert np.isclose(mtd.lambdas[1], 1., atol=0.01)
    assert np.isclose(mtd.transition_matrices[1][0, 0], 0.2, atol=0.01)
    assert np.isclose(mtd.transition_matrices[1][1, 1], 0.1, atol=0.01)
    assert np.isclose(mtd.transition_matrices[1][1, 0], 0.9, atol=0.01)
    assert np.isclose(mtd.transition_matrices[1][0, 1], 0.8, atol=0.01)
