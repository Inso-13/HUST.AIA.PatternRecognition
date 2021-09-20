import numpy as np


class inverse_method:
    def __init__(self):
        self.w = None

    def train(self, X, Y):
        X_p = np.dot(np.linalg.inv(np.dot(X.transpose(), X)), X.transpose())
        self.w = np.dot(X_p, Y)
        return self.w

    def test(self, X, Y):
        num_right = 0
        n = X.shape[0]
        for i in range(n):
            if np.sign(np.dot(self.w.transpose(), X[i])) == Y[i]:
                num_right += 1
        print("Inverse method accuracy: " + str(num_right / n))
