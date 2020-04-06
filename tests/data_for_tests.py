import numpy as np

data_for_tests = dict()

data_for_tests['transition_matrices'] = np.array([[[0.4, 0.6],
                                                   [0.7, 0.3]],
                                                  [[0.1, 0.9],
                                                   [0.5, 0.5]]])

data_for_tests['lambdas'] = np.array([0.4, 0.6])

data_for_tests['indexes'] = [(0, 0, 0),
                             (0, 0, 1),
                             (0, 1, 0),
                             (0, 1, 1),
                             (1, 0, 0),
                             (1, 0, 1),
                             (1, 1, 0),
                             (1, 1, 1)]

data_for_tests['expected_p_array'] = np.array([[8 / 11, 3 / 11],
                                               [12 / 39, 27 / 39],
                                               [8 / 23, 15 / 23],
                                               [12 / 27, 15 / 27],
                                               [14 / 17, 3 / 17],
                                               [2 / 11, 9 / 11],
                                               [14 / 29, 15 / 29],
                                               [2 / 7, 5 / 7]])
