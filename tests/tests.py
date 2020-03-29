from mtdlearn.mtd import MTD
from mtdlearn.preprocessing import parse_markov_matrix
import pytest
import numpy as np
import logging

logger = logging.getLogger(__name__)

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


def test_create_indexes():
    mtd = MTD(4, 3)
    assert len(mtd.indexes_) == 256


def test_ex_max():

    for seed in range(100):
        np.random.seed(seed)
        mtd = MTD(3, 2, verbose=0, number_of_initiations=1)
        mtd.fit(x, y)
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
    mtd.fit(x, y)
    assert mtd.transition_matrix.max() <= 1.0
    assert mtd.transition_matrix.min() >= 0.0
    assert np.isclose(mtd.transition_matrix.sum(1).max(), 1.0)
    assert mtd.transition_matrix.shape == (9, 3)


def test_init_method_error():

    with pytest.raises(ValueError):
        mtd = MTD(4, 3, init_method='a')


def test_parsing_input_type():
    with pytest.raises(TypeError):
        parse_markov_matrix([1, 2, 3])


def test_parsing_output_type():

    out = parse_markov_matrix(x)
    assert out.shape == (20,)
