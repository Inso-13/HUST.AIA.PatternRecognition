import matplotlib.pyplot as plt
from src.Logistic import *
from src.util import *

if __name__ == "__main__":
    lr = 0.01
    batch_size = 1
    epoch = 100
    size = 200
    mu1 = np.array([[-5, 0]])
    mu2 = np.array([[0, 5]])
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
    my_logistic = Logistic(train_x, train_y, lr, batch_size)
    my_logistic.train(epoch)
    my_logistic.test(test_x, test_y)

    x = np.linspace(-10, 5, 500)
    y = - my_logistic.w[0] / my_logistic.w[2] - my_logistic.w[1] / my_logistic.w[2] * x
    # plt.plot(x, y, label="Logistic")
    # plt.plot(*X1.T, '.', label='+1')
    # plt.plot(*X2.T, '+', label='-1')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.xlim(0, 100)
    plt.plot(my_logistic.losses)
    plt.show()
