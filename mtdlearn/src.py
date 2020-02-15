import numpy as np


class MTD:

    def __init__(self,
                 n_dimensions,
                 order,
                 lambdas_init='flat',
                 tmatrices_init='flat'):

        self.n_dimensions = n_dimensions
        self.order = order
        if lambdas_init == 'flat':
            self.lambdas_ = np.ones(order) / order
        if tmatrices_init == 'flat':
            self.tmatrices_ = np.ones((order, n_dimensions, n_dimensions)) / n_dimensions

