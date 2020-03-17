from mtdlearn import MTD
import pytest
import numpy as np


def test_create_indexes():
    mtd = MTD(4, 3)
    assert len(mtd.indexes) == 256


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
        mtd = MTD(3, 2, verbose=0, init_num=1)
        mtd.fit(np.random.randint(0, 100, 27, ))
        assert mtd.lambdas_.shape == (2, )
        assert np.isclose(sum(mtd.lambdas_), 1.0)
        assert max(mtd.lambdas_) <= 1
        assert min(mtd.lambdas_) >= 0
        assert np.isclose(sum(mtd.transition_matrices_[0, 0, :]), 1.0)
        assert mtd.transition_matrices_.shape == (2, 3, 3)
        assert mtd.transition_matrices_.min() >= 0
        assert mtd.transition_matrices_.max() <= 1


def test_create_markov():
    mtd = MTD(3, 2, verbose=0)
    mtd.fit(np.array([i for i in range(27)]))
    mtd.create_markov()
    assert mtd.transition_matrix_.max() <= 1.0
    assert mtd.transition_matrix_.min() >= 0.0
    assert np.isclose(mtd.transition_matrix_.sum(1).max(), 1.0)
    assert mtd.transition_matrix_.shape == (9, 3)


def test_aic():
    mtd1 = MTD(3, 2, verbose=0)
    mtd1.fit(np.array([i for i in range(27)]))
    mtd1._calculate_aic()
    mtd2 = MTD(3, 3, verbose=0)
    mtd2.fit(np.array([i for i in range(81)]))
    mtd2._calculate_aic()
    assert mtd1.aic < mtd2.aic
