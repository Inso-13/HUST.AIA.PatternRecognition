import numpy as np
import cvxopt
from src.util import *


class PrimalSVM:
    def __init__(self, X, Y):
        self.d = X.shape[1]
        self.A = np.ones_like(X)
        self.w = None
        self.Q = np.eye(self.d)
        self.Q[0][0] = 0
        self.p = np.zeros((self.d, 1))
        for i in range(self.A.shape[0]):
            self.A[i] = Y[i] * X[i]
        self.c = np.ones((self.A.shape[0], 1))

        self.Q = cvxopt.matrix(self.Q, tc='d')
        self.p = cvxopt.matrix(self.p, tc='d')
        self.A = cvxopt.matrix(self.A, tc='d')
        self.c = cvxopt.matrix(self.c, tc='d')

    def QP(self):
        ret = cvxopt.solvers.qp(self.Q, self.p, -self.A, -self.c)
        self.w = ret['x']
        return ret['x']

    def test(self, X, Y):
        num_right = 0
        n = X.shape[0]
        for i in range(n):
            if np.sign(np.dot(self.w.T, X[i])) == Y[i]:
                num_right += 1
        print("Test accuracy: " + str(num_right / n))
