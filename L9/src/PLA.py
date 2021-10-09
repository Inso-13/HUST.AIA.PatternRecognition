import numpy as np


class PLA:
    def __init__(self, train_x, train_y, test_x, test_y, max_times=10000, seed=0):
        self.max_times = max_times
        np.random.seed(seed)
        np.random.shuffle(train_x)
        np.random.seed(seed)
        np.random.shuffle(train_y)
        self.train_x = np.array(train_x)
        self.train_y = np.array(train_y)
        self.test_x = np.array(test_x)
        self.test_y = np.array(test_y)
        self.w = None

    def append1(self, x):
        return np.append(1, x)

    def train(self):
        times = 0
        PLA_is_over = False
        w = np.zeros(self.train_x.shape[1] + 1)
        n = self.train_x.shape[0]
        inputs = np.zeros((self.train_x.shape[0], self.train_x.shape[1] + 1))
        for i in range(n):
            inputs[i] = self.append1(self.train_x[i])
        while not PLA_is_over and times < self.max_times:
            PLA_is_over = True
            for i in range(n):
                pred_y = np.dot(w, inputs[i])
                signed_pred_y = np.sign(pred_y)
                if signed_pred_y != self.train_y[i]:
                    PLA_is_over = False
                    w += self.train_y[i] * inputs[i]
                    break
            times += 1
        self.w = w
        return w

    def test(self):
        num_right = 0
        n = self.test_x.shape[0]
        for i in range(n):
            if self.test_once(self.test_x[i], self.test_y[i]):
                num_right += 1
        print("accuracy: " + str(num_right / n))

    def test_once(self, x, y):
        return np.sign(self.w @ self.append1(x)) == y

    def test_pos_neg(self, x):
        return np.sign(self.w @ self.append1(x))