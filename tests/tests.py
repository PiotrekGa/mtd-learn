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


def test_input_len_noerror():
    mtd = MTD(3, 2, verbose=0)
    mtd.fit(np.array([i for i in range(27)]))


def test_ex_max():

    for seed in range(100):
        np.random.seed(seed)
        mtd = MTD(3, 2, verbose=0)
        mtd.fit(np.random.randint(0, 100, 27, ))
        assert mtd.lambdas_.shape == (2, )
        assert np.isclose(sum(mtd.lambdas_), 1.0)
        assert max(mtd.lambdas_) <= 1
        assert min(mtd.lambdas_) >= 0
        assert np.isclose(sum(mtd.transition_matrices_[0, 0, :]), 1.0)
        assert mtd.transition_matrices_.shape == (2, 3, 3)
        assert mtd.transition_matrices_.min() >= 0
        assert mtd.transition_matrices_.max() <= 1
