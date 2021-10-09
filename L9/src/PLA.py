import numpy as np


class PLA:
    def __init__(self, inputs, labels, train_data_rate=0.8, max_times=10000, seed=0):
        self.train_data_rate = train_data_rate
        self.max_times = max_times
        num_of_inputs = inputs.shape[0]
        np.random.seed(seed)
        np.random.shuffle(inputs)
        np.random.seed(seed)
        np.random.shuffle(labels)
        self.train_x = inputs[0:int(num_of_inputs * self.train_data_rate)]
        self.train_y = labels[0:int(num_of_inputs * self.train_data_rate)]
        self.test_x = inputs[int(num_of_inputs * self.train_data_rate):]
        self.test_y = labels[int(num_of_inputs * self.train_data_rate):]
        self.w = None

    def append1(self, x):
        return np.append(1, x)

    def sign(self, x):
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0

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
                signed_pred_y = self.sign(pred_y)
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
            if self.sign(np.dot(self.w, self.append1(self.test_x[i]))) == self.test_y[i]:
                num_right += 1
        print("accuracy: " + str(num_right / n))
