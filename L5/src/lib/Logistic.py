from .optimizer import *


class Logistic:
    def __init__(self, X, Y, lr=0.4, batch_size=1):
        self.lr = lr
        self.X = X
        self.Y = Y
        self.w = np.zeros((self.X.shape[-1], self.Y.shape[-1]))
        self.SGD = SGD(self.w, self.grad_fn, self.X, self.Y, lr, batch_size)
        self.losses = []
        self.N = self.X.shape[0]

    def step(self):
        self.w = self.SGD.step()
        self.losses.append(self.CrossEntropyLoss(self.w, self.X, self.Y))

    def train(self, epochs=100):
        for i in range(epochs):
            self.step()

    def test(self, X, Y):
        for i in range(X.shape[0]):
            probability = self.Sigmoid(self.w.T @ X[i])
            if Y[i] == -1:
                probability = 1 - probability

            print("Sample ID: {}, Class: {}, Probability: {}".format(i, Y[i][0], probability))

    @staticmethod
    def grad_fn(w, x, y):
        y = y.reshape(-1, 1)
        x = x.reshape(-1, 1)
        return Logistic.Sigmoid(-y @ w.T @ x) * -y * x

    @staticmethod
    def CrossEntropyLoss(w, X, Y):
        loss = 0
        n = X.shape[0]
        for i in range(n):
            loss += np.log(1 + np.exp(-Y[i] * w.T @ X[i].T.reshape(X.shape[-1], 1)))
        loss /= n
        return loss[0]

    @staticmethod
    def Sigmoid(X):
        return np.exp(X) / (1 + np.exp(X))
