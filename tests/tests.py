from mtdlearn import MTD
import pytest
import numpy as np


def test_create_lambdas():
    mtd = MTD(5, 4, lambdas_init='flat')
    assert len(mtd.lambdas_) == 4
    assert max(mtd.lambdas_) == min(mtd.lambdas_)
    assert mtd.lambdas_[0] == 0.25
