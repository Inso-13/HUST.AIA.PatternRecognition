import matplotlib.pyplot as plt
from numpy.core.numeric import ones_like
from lib.KernelSVM import *
from lib.util import *


def guass_kernel_function(x_i):
    if len(x_i.shape) == 1:
        return np.exp(-(x_i[0] ** 2 + x_i[1] ** 2)) * np.array(
            [1, x_i[0], x_i[1], x_i[0] * x_i[0], x_i[0] * x_i[1], x_i[1] * x_i[1],
             x_i[0] * x_i[0] * x_i[0], x_i[0] * x_i[0] * x_i[1], x_i[0] * x_i[1] * x_i[1],
             x_i[1] * x_i[1] * x_i[1]])

    elif len(x_i.shape) == 3:
        shape1, shape2 = x_i.shape[1:]
        ret = np.zeros((10, shape1, shape2))
        for i in range(shape1):
            for j in range(shape2):
                ret[:, i, j] = np.exp(-(x_i[0][i][j] ** 2 + x_i[1][i][j] ** 2)) * np.array(
                    [1, x_i[0][i][j], x_i[1][i][j], x_i[0][i][j] * x_i[0][i][j], x_i[0][i][j] * x_i[1][i][j],
                     x_i[1][i][j] * x_i[1][i][j],
                     x_i[0][i][j] * x_i[0][i][j] * x_i[0][i][j], x_i[0][i][j] * x_i[0][i][j] * x_i[1][i][j],
                     x_i[0][i][j] * x_i[1][i][j] * x_i[1][i][j], x_i[1][i][j] * x_i[1][i][j] * x_i[1][i][j]])
        return ret


def line2_kernel_function(x_i):
    if len(x_i.shape) == 1:
        return np.array([x_i[0], x_i[1]])
    elif len(x_i.shape) == 3:
        shape1, shape2 = x_i.shape[1:]
        ret = np.zeros((2, shape1, shape2))
        for i in range(shape1):
            for j in range(shape2):
                ret[:, i, j] = np.array([x_i[0][i][j], x_i[1][i][j]])
        return ret


def line4_kernel_function(x_i):
    if len(x_i.shape) == 1:
        return np.array([1, x_i[0], x_i[1], x_i[0] * x_i[0], x_i[0] * x_i[1], x_i[1] * x_i[1],
                         x_i[0] * x_i[0] * x_i[0], x_i[0] * x_i[0] * x_i[1], x_i[0] * x_i[1] * x_i[1],
                         x_i[1] * x_i[1] * x_i[1],
                         x_i[0] * x_i[0] * x_i[0] * x_i[0], x_i[0] * x_i[0] * x_i[0] * x_i[1],
                         x_i[0] * x_i[0] * x_i[1] * x_i[1],
                         x_i[0] * x_i[1] * x_i[1] * x_i[1], x_i[1] * x_i[1] * x_i[1] * x_i[1]])

    elif len(x_i.shape) == 3:
        shape1, shape2 = x_i.shape[1:]
        ret = np.zeros((15, shape1, shape2))
        for i in range(shape1):
            for j in range(shape2):
                ret[:, i, j] = np.array(
                    [1, x_i[0][i][j], x_i[1][i][j], x_i[0][i][j] * x_i[0][i][j], x_i[0][i][j] * x_i[1][i][j],
                     x_i[1][i][j] * x_i[1][i][j],
                     x_i[0][i][j] * x_i[0][i][j] * x_i[0][i][j], x_i[0][i][j] * x_i[0][i][j] * x_i[1][i][j],
                     x_i[0][i][j] * x_i[1][i][j] * x_i[1][i][j], x_i[1][i][j] * x_i[1][i][j] * x_i[1][i][j],
                     x_i[0][i][j] * x_i[0][i][j] * x_i[0][i][j] * x_i[0][i][j],
                     x_i[0][i][j] * x_i[0][i][j] * x_i[0][i][j] * x_i[1][i][j],
                     x_i[0][i][j] * x_i[0][i][j] * x_i[1][i][j] * x_i[1][i][j],
                     x_i[0][i][j] * x_i[1][i][j] * x_i[1][i][j] * x_i[1][i][j],
                     x_i[1][i][j] * x_i[1][i][j] * x_i[1][i][j] * x_i[1][i][j]])
        return ret


kernel_function = line2_kernel_function
if __name__ == "__main__":
    X1 = np.array([[119.28, 26.08],
                   [121.31, 25.03],
                   [121.47, 31.23],
                   [118.06, 24.27],
                   [121.46, 39.04],
                   [122.10, 37.50],
                   [124.23, 40.07]])

    X2 = np.array([[129.87, 32.75],
                   [130.33, 31.36],
                   [131.42, 31.91],
                   [130.24, 33.35],
                   [133.33, 15.43],
                   [138.38, 34.98],
                   [140.47, 36.37]])

    # X1 = np.array([[119.28, 26.08],
    #                [121.31, 25.03],
    #                [121.47, 31.23],
    #                [118.06, 24.27],
    #                [113.53, 29.58],
    #                [104.06, 30.67],
    #                [116.25, 39.54],
    #                [121.46, 39.04],
    #                [122.10, 37.50],
    #                [124.23, 40.07]])
    #
    # X2 = np.array([[129.87, 32.75],
    #                [130.33, 31.36],
    #                [131.42, 31.91],
    #                [130.24, 33.35],
    #                [136.54, 35.10],
    #                [132.27, 34.24],
    #                [139.46, 35.42],
    #                [133.33, 15.43],
    #                [138.38, 34.98],
    #                [140.47, 36.37]])

    Y1 = np.ones((X1.shape[0], 1))
    Y2 = -np.ones((X2.shape[0], 1))
    X = np.concatenate((X1, X2))
    X = add_first_1_for_x(X)
    Y = np.concatenate((Y1, Y2))
    # train_x, test_x, train_y, test_y = divide_dataset(X, Y, 0.8)
    train_x, train_y = X, Y
    test_x, test_y = np.array([[1, 123.28, 25.45]]), np.array([[1]])
    my_KernelSVM = KernelSVM(train_x, train_y, kernel_function)
    my_KernelSVM.QP()
    my_KernelSVM.test(train_x, train_y)
    my_KernelSVM.test(test_x, test_y)

    xx = np.arange(110, 140, 0.01)
    yy = np.arange(20, 40, 0.01)
    xx, yy = np.meshgrid(xx, yy)
    re1 = my_KernelSVM.w[1:].reshape((1, len(my_KernelSVM.w[1:])))
    re2 = kernel_function(np.array([xx, yy]))
    z = (re1 @ re2.reshape(re2.shape[0], re2.shape[1] * re2.shape[2]) + my_KernelSVM.w[0]).reshape(
        (re2.shape[1], re2.shape[2]))
    plt.contour(xx, yy, z, 0)

    # plt.plot(x, y, label="KernelSVM")
    plt.plot(*X1.T, '.', label='+1')
    plt.plot(*X2.T, '+', label='-1')
    plt.scatter([test_x[0][1]], [test_x[0][2]])

    # X1_tx = []
    # X1_ty = []
    # X2_tx = []
    # X2_ty = []
    # for this_x in X1:
    #     if abs(my_KernelSVM.w[1:].T @ kernel_function(this_x) + my_KernelSVM.w[0] - 1) < 1e-3:
    #         X1_tx.append(this_x[0])
    #         X1_ty.append(this_x[1])
    # for this_x in X2:
    #     if abs(my_KernelSVM.w[1:].T @ kernel_function(this_x) + my_KernelSVM.w[0] + 1) < 1e-3:
    #         X2_tx.append(this_x[0])
    #         X2_ty.append(this_x[1])
    # if len(X1_tx):
    #     plt.scatter(X1_tx, X1_ty)
    # if len(X2_tx):
    #     plt.scatter(X2_tx, X2_ty)
    plt.show()
