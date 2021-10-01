from src.KernelSVM import *


class DualSVM(KernelSVM):
    def __init__(self, X, Y):
        super().__init__(X, Y, self.kernel_function)

    @staticmethod
    def kernel_function(x_i):
        if len(x_i.shape) == 1:
            return np.array([1, x_i[0], x_i[1]])
        elif len(x_i.shape) == 3:
            shape1, shape2 = x_i.shape[1:]
            ret = np.zeros((3, shape1, shape2))
            for i in range(shape1):
                for j in range(shape2):
                    ret[:, i, j] = np.array([1, x_i[0][i][j], x_i[1][i][j]])
            return ret