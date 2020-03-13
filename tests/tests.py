from mtdlearn import MTD
import pytest
import numpy as np


def test_create_lambdas():
    mtd = MTD(5, 4, lambdas_init='flat')
    assert len(mtd.lambdas_) == 4
    assert max(mtd.lambdas_) == min(mtd.lambdas_)
    assert mtd.lambdas_[0] == 0.25


def test_create_tmatrices():
    mtd = MTD(5, 4, tmatrices_init='flat')
    assert mtd.tmatrices_.shape == (4, 5, 5)
    assert mtd.tmatrices_[0, 0, 0] == 0.2


def test_create_markov():
    mtd = MTD(5, 4)
    mtd.create_markov()
    assert mtd.transition_matrix_.shape == (625, 5)


def test_create_indexes():
    mtd = MTD(4, 3)
    assert len(mtd.indexes) == 256


def test_input_len_error():
    
    with pytest.raises(ValueError):
        mtd = MTD(4, 3)
        mtd.fit(np.array([1, 2, 3]))


def test_input_len_noerror():
    mtd = MTD(3, 2)
    mtd.fit(np.array([i for i in range(27)]))


def test_input_probs():
    mtd = MTD(2, 2)
    mtd.fit(np.array([1 for _ in range(8)]))
    assert mtd.p_[0] == 0.125
    assert mtd.p_.shape == (8,)
    assert mtd.n_.shape == (8,)
    assert mtd.n_direct_[0, 0, 0] ==2