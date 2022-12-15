import matplotlib.pyplot as plt
from lib.PrimalSVM import *
from lib.util import *

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
    my_PrimalSVM = PrimalSVM(train_x, train_y)
    my_PrimalSVM.QP()
    my_PrimalSVM.test(train_x, train_y)
    my_PrimalSVM.test(test_x, test_y)

    x = np.linspace(-10, 10, 500)
    y = - my_PrimalSVM.w[0] / my_PrimalSVM.w[2] - my_PrimalSVM.w[1] / my_PrimalSVM.w[2] * x
    plt.plot(x, y, label="PrimalSVM")
    plt.plot(*X1.T, '.', label='+1')
    plt.plot(*X2.T, '+', label='-1')

    X1_tx = []
    X1_ty = []
    X2_tx = []
    X2_ty = []
    for this_x in X1:
        if abs(my_PrimalSVM.w[1:].T @ this_x + my_PrimalSVM.w[0] - 1) < 1e-3:
            X1_tx.append(this_x[0])
            X1_ty.append(this_x[1])
    for this_x in X2:
        if abs(my_PrimalSVM.w[1:].T @ this_x + my_PrimalSVM.w[0] + 1) < 1e-3:
            X2_tx.append(this_x[0])
            X2_ty.append(this_x[1])
    if len(X1_tx):
        plt.scatter(X1_tx, X1_ty)
    if len(X2_tx):
        plt.scatter(X2_tx, X2_ty)
    plt.show()
