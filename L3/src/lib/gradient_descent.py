import numpy as np


class GD:
    def __init__(self, shape, lr):
        self.w = np.zeros(shape)
        self.lr = lr
        self.losses = []

    def step(self, X, Y):
        N = X.shape[0]
        self.w -= self.lr * 2 / N * (np.dot(np.dot(X.transpose(), X), self.w) - np.dot(X.transpose(), Y))
        self.losses.append(np.linalg.norm(np.dot(X, self.w) - Y) / N)

    def train(self, X, Y, epochs=100):
        for i in range(epochs):
            self.step(X, Y)

    def test(self, X, Y):
        num_right = 0
        n = X.shape[0]
        for i in range(n):
            if np.sign(np.dot(self.w.transpose(), X[i])) == Y[i]:
                num_right += 1
        print("Gradient descent accuracy: " + str(num_right / n))
