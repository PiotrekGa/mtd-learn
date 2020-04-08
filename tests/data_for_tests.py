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

expected_p_direct_array = np.array(
    [[[data_for_tests['expected_p_array'][0, 0] + data_for_tests['expected_p_array'][2, 0],
       data_for_tests['expected_p_array'][1, 0] + data_for_tests['expected_p_array'][3, 0]],
      [data_for_tests['expected_p_array'][4, 0] + data_for_tests['expected_p_array'][6, 0],
       data_for_tests['expected_p_array'][5, 0] + data_for_tests['expected_p_array'][7, 0]]],
     [[data_for_tests['expected_p_array'][0, 1] + data_for_tests['expected_p_array'][4, 1],
       data_for_tests['expected_p_array'][1, 1] + data_for_tests['expected_p_array'][5, 1]],
      [data_for_tests['expected_p_array'][2, 1] + data_for_tests['expected_p_array'][6, 1],
       data_for_tests['expected_p_array'][3, 1] + data_for_tests['expected_p_array'][7, 1]]]
     ])
data_for_tests['expected_p_direct_array'] = expected_p_direct_array / expected_p_direct_array.sum(0)

expected_p_direct_array2 = expected_p_direct_array.sum(2)
data_for_tests['expected_p_direct_array2'] = expected_p_direct_array2 / expected_p_direct_array2.sum(1).reshape(-1, 1)

data_for_tests['n_passes'] = np.array([100, 100, 400, 400, 100, 100, 400, 400])

data_for_tests['n_passes_direct'] = np.array([[[500, 500],
                                               [500, 500]],
                                              [[200, 200],
                                               [800, 800]]])

denominator = 1 / (data_for_tests['n_passes'].sum() - 2)
lambda1 = denominator * np.dot(data_for_tests['expected_p_array'][:, 0], data_for_tests['n_passes'])
lambda2 = denominator * np.dot(data_for_tests['expected_p_array'][:, 1], data_for_tests['n_passes'])

lambda_sum = lambda1 + lambda2

lambda1 = lambda1 / lambda_sum
lambda2 = lambda2 / lambda_sum

data_for_tests['expected_lambdas'] = np.array([lambda1, lambda2])

expected_transition_matrices = data_for_tests['n_passes_direct'] * data_for_tests['expected_p_direct_array']
expected_transition_matrices = expected_transition_matrices / expected_transition_matrices.sum(2).reshape(2, -1, 1)

data_for_tests['expected_transition_matrices'] = expected_transition_matrices

log_likelihood1 = 0

log_likelihood1 = log_likelihood1 + data_for_tests['n_passes'][0] * np.log(
    data_for_tests['lambdas'][0] * data_for_tests['transition_matrices'][0, 0, 0] +
    data_for_tests['lambdas'][1] * data_for_tests['transition_matrices'][1, 0, 0])

log_likelihood1 = log_likelihood1 + data_for_tests['n_passes'][1] * np.log(
    data_for_tests['lambdas'][0] * data_for_tests['transition_matrices'][0, 0, 1] +
    data_for_tests['lambdas'][1] * data_for_tests['transition_matrices'][1, 0, 1])

log_likelihood1 = log_likelihood1 + data_for_tests['n_passes'][2] * np.log(
    data_for_tests['lambdas'][0] * data_for_tests['transition_matrices'][0, 0, 0] +
    data_for_tests['lambdas'][1] * data_for_tests['transition_matrices'][1, 1, 0])

log_likelihood1 = log_likelihood1 + data_for_tests['n_passes'][3] * np.log(
    data_for_tests['lambdas'][0] * data_for_tests['transition_matrices'][0, 0, 1] +
    data_for_tests['lambdas'][1] * data_for_tests['transition_matrices'][1, 1, 1])

log_likelihood1 = log_likelihood1 + data_for_tests['n_passes'][4] * np.log(
    data_for_tests['lambdas'][0] * data_for_tests['transition_matrices'][0, 1, 0] +
    data_for_tests['lambdas'][1] * data_for_tests['transition_matrices'][1, 0, 0])

log_likelihood1 = log_likelihood1 + data_for_tests['n_passes'][5] * np.log(
    data_for_tests['lambdas'][0] * data_for_tests['transition_matrices'][0, 1, 1] +
    data_for_tests['lambdas'][1] * data_for_tests['transition_matrices'][1, 0, 1])

log_likelihood1 = log_likelihood1 + data_for_tests['n_passes'][6] * np.log(
    data_for_tests['lambdas'][0] * data_for_tests['transition_matrices'][0, 1, 0] +
    data_for_tests['lambdas'][1] * data_for_tests['transition_matrices'][1, 1, 0])

log_likelihood1 = log_likelihood1 + data_for_tests['n_passes'][7] * np.log(
    data_for_tests['lambdas'][0] * data_for_tests['transition_matrices'][0, 1, 1] +
    data_for_tests['lambdas'][1] * data_for_tests['transition_matrices'][1, 1, 1])

data_for_tests['log_likelihood1'] = log_likelihood1

log_likelihood2 = 0

log_likelihood2 = log_likelihood2 + data_for_tests['n_passes'][0] * np.log(
    data_for_tests['expected_lambdas'][0] * data_for_tests['expected_transition_matrices'][0, 0, 0] +
    data_for_tests['expected_lambdas'][1] * data_for_tests['expected_transition_matrices'][1, 0, 0])

log_likelihood2 = log_likelihood2 + data_for_tests['n_passes'][1] * np.log(
    data_for_tests['expected_lambdas'][0] * data_for_tests['expected_transition_matrices'][0, 0, 1] +
    data_for_tests['expected_lambdas'][1] * data_for_tests['expected_transition_matrices'][1, 0, 1])

log_likelihood2 = log_likelihood2 + data_for_tests['n_passes'][2] * np.log(
    data_for_tests['expected_lambdas'][0] * data_for_tests['expected_transition_matrices'][0, 0, 0] +
    data_for_tests['expected_lambdas'][1] * data_for_tests['expected_transition_matrices'][1, 1, 0])

log_likelihood2 = log_likelihood2 + data_for_tests['n_passes'][3] * np.log(
    data_for_tests['expected_lambdas'][0] * data_for_tests['expected_transition_matrices'][0, 0, 1] +
    data_for_tests['expected_lambdas'][1] * data_for_tests['expected_transition_matrices'][1, 1, 1])

log_likelihood2 = log_likelihood2 + data_for_tests['n_passes'][4] * np.log(
    data_for_tests['expected_lambdas'][0] * data_for_tests['expected_transition_matrices'][0, 1, 0] +
    data_for_tests['expected_lambdas'][1] * data_for_tests['expected_transition_matrices'][1, 0, 0])

log_likelihood2 = log_likelihood2 + data_for_tests['n_passes'][5] * np.log(
    data_for_tests['expected_lambdas'][0] * data_for_tests['expected_transition_matrices'][0, 1, 1] +
    data_for_tests['expected_lambdas'][1] * data_for_tests['expected_transition_matrices'][1, 0, 1])

log_likelihood2 = log_likelihood2 + data_for_tests['n_passes'][6] * np.log(
    data_for_tests['expected_lambdas'][0] * data_for_tests['expected_transition_matrices'][0, 1, 0] +
    data_for_tests['expected_lambdas'][1] * data_for_tests['expected_transition_matrices'][1, 1, 0])

log_likelihood2 = log_likelihood2 + data_for_tests['n_passes'][7] * np.log(
    data_for_tests['expected_lambdas'][0] * data_for_tests['expected_transition_matrices'][0, 1, 1] +
    data_for_tests['expected_lambdas'][1] * data_for_tests['expected_transition_matrices'][1, 1, 1])

data_for_tests['log_likelihood2'] = log_likelihood2
