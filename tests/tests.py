from mtdlearn.mtd import MTD
from mtdlearn.preprocessing import PathEncoder
from mtdlearn.datasets import data_values3_order2_full as data
from mtdlearn.datasets import generate_data
import pytest
import numpy as np
import logging

logger = logging.getLogger(__name__)

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
    x_gen, y_gen = generate_data(('A', 'B', 'C', 'D'), '>', 5, 5, 5, 100, 1.0)
    assert x_gen.shape[0] == y_gen.shape[0]
    assert y_gen.shape[0] == 100
    assert max([len(i[0].split('>')) for i in x_gen]) == 5
    assert min([len(i[0].split('>')) for i in x_gen]) == 5
    for i, row in enumerate(x_gen):
        assert row[0].split('>')[0] == y_gen[i]


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
