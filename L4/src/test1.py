from lib.Fisher import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    X1 = np.array([[5, 37], [7, 30], [10, 35], [11.5, 40], [14, 38], [12, 31]])
    Y1 = np.array([1, 1, 1, 1, 1, 1])
    X2 = np.array([[35, 21.5], [39, 21.7], [34, 16], [37, 17]])
    Y2 = np.array([-1, -1, -1, -1])
    my_fisher = Fisher()
    my_fisher.train((X1, Y1), (X2, Y2))

    X = np.concatenate((X1, X2))
    Y = np.concatenate((Y1, Y2))

    x = np.linspace(-10, 50, 500)
    y = my_fisher.y0 / my_fisher.w[1][0] - my_fisher.w[0][0] / my_fisher.w[1][0] * x
    plt.plot(x, y[0], label="Fisher")
    plt.plot(*X1.T, '.', label='+1')
    plt.plot(*X2.T, '+', label='-1')

    plt.show()
