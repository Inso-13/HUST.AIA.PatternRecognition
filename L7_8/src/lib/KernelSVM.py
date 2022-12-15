import numpy as np
import cvxopt
from .util import *


class KernelSVM:
    def __init__(self, X, Y, kernel_function):
        self.d = len(kernel_function(np.array([0, 0]))) + 1
        self.kernel_function = kernel_function
        self.X = X
        self.Y = Y
        self.A = np.ones_like(X)
        self.w = np.zeros(self.d)
        self.Q = np.ones((X.shape[0], X.shape[0]))
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                self.Q[i][j] = Y[i] * Y[j] * self.kernel_function(X[i][1:]).T @ self.kernel_function(X[j][1:])
        self.p = -np.ones((self.A.shape[0], 1))
        self.u = None
        self.c = np.zeros((self.A.shape[0], 1))
        self.v = 0
        self.A = np.eye(X.shape[0])
        self.r = Y.T
        self.alpha = None

        self.Q = cvxopt.matrix(self.Q, tc='d')
        self.p = cvxopt.matrix(self.p, tc='d')
        self.A = cvxopt.matrix(self.A, tc='d')
        self.c = cvxopt.matrix(self.c, tc='d')
        self.r = cvxopt.matrix(self.r, tc='d')
        self.v = cvxopt.matrix(self.v, tc='d')

    def QP(self):
        ret = cvxopt.solvers.qp(self.Q, self.p, -self.A, -self.c, self.r, self.v)
        self.alpha = ret['x']
        max_alpha = -1
        j = -1
        for i in range(self.X.shape[0]):
            self.w[1:] += self.alpha[i] * self.Y[i] * self.kernel_function(self.X[i][1:])
            if self.alpha[i] > max_alpha:
                max_alpha = self.alpha[i]
                j = i
        temp_sum = 0
        for i in range(self.X.shape[0]):
            temp_sum += self.alpha[i] * self.Y[i] * self.kernel_function(self.X[j][1:]).T @ self.kernel_function(self.X[i][1:])
        self.w[0] = self.Y[j] - temp_sum
        return self.w

    def test(self, X, Y):
        num_right = 0
        n = X.shape[0]
        for i in range(n):
            temp_sum = 0
            for j in range(self.X.shape[0]):
                temp_sum += self.alpha[j] * self.Y[j] * self.kernel_function(self.X[j][1:]).T @ self.kernel_function(
                    X[i][1:])
            if np.sign(temp_sum + self.w[0]) == Y[i]:
                num_right += 1
        print("Test accuracy: " + str(num_right / n))
