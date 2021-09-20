import numpy as np
import matplotlib.pyplot as plt
from src.inverse import *
from src.gradient_descent import *
from src.util import *


if __name__ == "__main__":
    mu1 = np.array([[1, 0]])
    mu2 = np.array([[0, 1]])
    Sigma = np.array([[1, 0], [0, 1]])
    R = np.linalg.cholesky(Sigma).T
    X1 = np.random.randn(200, 2) @ R + mu1
    X2 = np.random.randn(200, 2) @ R + mu2
    Y1 = np.ones((200, 1))
    Y2 = -Y1
    X = np.concatenate((X1, X2))
    X = add_first_1_for_x(X)
    Y = np.concatenate((Y1, Y2))
    train_x, test_x, train_y, test_y = divide_dataset(X, Y, 0.8)
    plt.subplot(131)
    my_inverse_method = inverse_method()
    my_inverse_method.train(train_x, train_y)
    my_inverse_method.test(train_x, train_y)
    x = np.linspace(-10, 5, 500)
    y = - my_inverse_method.w[0] / my_inverse_method.w[2] - my_inverse_method.w[1] / my_inverse_method.w[2] * x
    plt.plot(x, y, label="inverse_method")
    plt.plot(*X1.T, '.', label='+1')
    plt.plot(*X2.T, '+', label='-1')

    plt.subplot(132)
    my_GD = GD((3, 1), 0.04)
    my_GD.train(train_x, train_y, 100)
    my_GD.test(train_x, train_y)
    y = - my_GD.w[0] / my_GD.w[2] - my_GD.w[1] / my_GD.w[2] * x
    plt.plot(x, y, label="Gradient_descent")
    plt.plot(*X1.T, '.', label='+1')
    plt.plot(*X2.T, '+', label='-1')

    plt.subplot(133)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.ylim(0.04, 0.06)
    plt.plot(my_GD.losses)
    plt.legend()

    plt.show()
