from mtdlearn.mtd import MTD
from mtdlearn.preprocessing import parse_markov_matrix, PathEncoder
from mtdlearn.datasets import second_order
import pytest
import numpy as np
import logging

logger = logging.getLogger(__name__)

x = second_order['x']
y = second_order['y']
sample_weight = second_order['sample_weight']


def test_dataset():
    assert x.shape[0] == y.shape[0]
    assert x.shape[0] == sample_weight.shape[0]


def test_create_indexes():
    mtd = MTD(4, 3)
    assert len(mtd.indexes_) == 256


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


def test_init_method_error():

    with pytest.raises(ValueError):
        mtd = MTD(4, 3, init_method='a')


def test_parsing_input_type():
    with pytest.raises(TypeError):
        parse_markov_matrix([1, 2, 3])


def test_parsing_output_type():

    x_ones = np.ones((81, 3))
    out = parse_markov_matrix(x_ones)
    assert out.shape == (243,)
