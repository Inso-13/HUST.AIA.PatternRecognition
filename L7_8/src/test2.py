import matplotlib.pyplot as plt
from src.DualSVM import *
from src.util import *

if __name__ == "__main__":
    size = 200
    mu1 = np.array([[3, 0]])
    mu2 = np.array([[0, 3]])
    Sigma = np.array([[1, 0], [0, 1]])
    R = np.linalg.cholesky(Sigma).T
    X1 = np.random.randn(size, 2) @ R + mu1
    X2 = np.random.randn(size, 2) @ R + mu2
    Y1 = np.ones((size, 1))
    Y2 = -Y1
    X = np.concatenate((X1, X2))
    X = add_first_1_for_x(X)
    Y = np.concatenate((Y1, Y2))
    train_x, test_x, train_y, test_y = divide_dataset(X, Y, 0.8)
    my_DualSVM = DualSVM(train_x, train_y)
    my_DualSVM.QP()
    my_DualSVM.test(train_x, train_y)
    my_DualSVM.test(test_x, test_y)

    xx = np.arange(-10, 10, 0.01)
    yy = np.arange(-10, 10, 0.01)
    xx, yy = np.meshgrid(xx, yy)
    re1 = my_DualSVM.w[1:].reshape((1, len(my_DualSVM.w[1:])))
    re2 = my_DualSVM.kernel_function(np.array([xx, yy]))
    z = (re1 @ re2.reshape(re2.shape[0], re2.shape[1] * re2.shape[2])).reshape((re2.shape[1], re2.shape[2]))
    # z = (my_KernelSVM.w[1:].T @ kernel_function(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)
    plt.contour(xx, yy, z, 0)

    plt.plot(*X1.T, '.', label='+1')
    plt.plot(*X2.T, '+', label='-1')

    X1_tx = []
    X1_ty = []
    X2_tx = []
    X2_ty = []
    for this_x in X1:
        if abs(my_DualSVM.w[1:].T @ my_DualSVM.kernel_function(this_x) + my_DualSVM.w[0] - 1) < 1e-3:
            X1_tx.append(this_x[0])
            X1_ty.append(this_x[1])
    for this_x in X2:
        if abs(my_DualSVM.w[1:].T @ my_DualSVM.kernel_function(this_x) + my_DualSVM.w[0] + 1) < 1e-3:
            X2_tx.append(this_x[0])
            X2_ty.append(this_x[1])
    if len(X1_tx):
        plt.scatter(X1_tx, X1_ty)
    if len(X2_tx):
        plt.scatter(X2_tx, X2_ty)
    plt.show()
