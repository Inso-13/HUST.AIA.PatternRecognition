import numpy as np


def add_first_1_for_x(X):
    n = X.shape[0]
    ones = np.ones((n, 1))
    return np.concatenate((ones, X), 1)


def divide_dataset(X, Y, train_data_rate=0.8, seed=0):
    num_of_inputs = X.shape[0]
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(Y)
    train_x = X[0:int(num_of_inputs * train_data_rate)]
    train_y = Y[0:int(num_of_inputs * train_data_rate)]
    test_x = X[int(num_of_inputs * train_data_rate):]
    test_y = Y[int(num_of_inputs * train_data_rate):]
    return train_x, test_x, train_y, test_y
