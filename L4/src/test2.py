import matplotlib.pyplot as plt
from src.Fisher import *
from src.util import *

if __name__ == "__main__":
    mu1 = np.array([[-5, 0]])
    mu2 = np.array([[0, 5]])
    Sigma = np.array([[1, 0], [0, 1]])
    R = np.linalg.cholesky(Sigma).T
    X1 = np.random.randn(200, 2) @ R + mu1
    X2 = np.random.randn(200, 2) @ R + mu2
    Y1 = np.ones((200, 1))
    Y2 = -Y1
    train_x1, test_x1, train_y1, test_y1 = divide_dataset(X1, Y1, 0.8, 1)
    train_x2, test_x2, train_y2, test_y2 = divide_dataset(X2, Y2, 0.8, 1)
    test_x = np.concatenate((test_x1, test_x2))
    test_y = np.concatenate((test_y1, test_y2))
    my_fisher = Fisher()
    my_fisher.train((train_x1, train_y1), (train_x2, train_y2))
    my_fisher.test(test_x, test_y)

    X = np.concatenate((X1, X2))
    Y = np.concatenate((Y1, Y2))

    x = np.linspace(-10, 5, 500)
    y = my_fisher.y0 / my_fisher.w[1] - my_fisher.w[0] / my_fisher.w[1] * x
    plt.plot(x, y[0], label="Fisher")
    plt.plot(0, (my_fisher.y0 / my_fisher.w[1])[0], '*')
    y = x * (my_fisher.w[1] / my_fisher.w[0])[0] + (my_fisher.y0 / my_fisher.w[1])[0]
    plt.plot(x, y, label="Vector")
    plt.plot(*X1.T, '.', label='+1')
    plt.plot(*X2.T, '+', label='-1')

    plt.show()
