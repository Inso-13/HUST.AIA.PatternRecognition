import numpy as np


class Fisher:
    def __init__(self):
        self.lenX = 0
        self.w = None
        self.y0 = None

    def train(self, XY1, XY2):
        X1 = XY1[0]
        Y1 = XY1[1]
        X2 = XY2[0]
        Y2 = XY2[1]
        self.lenX = X1.shape[1]
        u1 = np.average(X1, axis=0)
        u2 = np.average(X2, axis=0)
        s1 = np.zeros((self.lenX, self.lenX))
        s2 = np.zeros((self.lenX, self.lenX))
        for i in range(X1.shape[0]):
            s1 = np.add(((X1[i] - u1).reshape(self.lenX, 1) @
                        (X1[i] - u1).reshape(1, self.lenX)), s1)
        for i in range(X2.shape[0]):
            s2 = np.add(((X2[i] - u2).reshape(self.lenX, 1) @
                        (X2[i] - u2).reshape(1, self.lenX)), s2)
        sw = s1 + s2
        self.w = np.linalg.inv(sw) @ (u1 - u2).reshape(self.lenX, 1)
        self.y0 = self.w.T @ (u1 + u2).reshape(self.lenX, 1) / 2

        num_right = 0
        n1 = X1.shape[0]
        for i in range(n1):
            if np.sign(np.dot(self.w.T, X1[i]) - self.y0) == Y1[i]:
                num_right += 1
        n2 = X2.shape[0]
        for i in range(n2):
            if np.sign(np.dot(self.w.T, X2[i]) - self.y0) == Y2[i]:
                num_right += 1
        print("Train accuracy: " + str(num_right / (n1 + n2)))

    def test(self, X, Y):
        num_right = 0
        n = X.shape[0]
        for i in range(n):
            if np.sign(np.dot(self.w.T, X[i]) - self.y0) == Y[i]:
                num_right += 1
        print("Test accuracy: " + str(num_right / n))
