import datetime
import numpy as np
import matplotlib.pyplot as plt


class Pocket:
    def __init__(self, inputs, labels, train_data_rate=0.8, max_times=1000, seed=0):
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
        self.new_train_x = None

    def append1(self, x):
        return np.append(1, x)

    def sign(self, x):
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0

    def num_error(self, w):
        ans = 0
        n = self.new_train_x.shape[0]
        for i in range(n):
            if self.sign(np.dot(w, self.new_train_x[i])) != self.train_y[i]:
                ans += 1
        return ans

    def train(self):
        times = 0
        w = np.zeros(self.train_x.shape[1] + 1)
        best_w = np.zeros(self.train_x.shape[1] + 1)
        n = self.train_x.shape[0]
        min_num_error = n + 1
        self.new_train_x = np.zeros(
            (self.train_x.shape[0], self.train_x.shape[1] + 1))
        for i in range(n):
            self.new_train_x[i] = self.append1(self.train_x[i])

        while times < self.max_times:
            for i in range(n):
                pred_y = np.dot(w, self.new_train_x[i])
                signed_pred_y = self.sign(pred_y)
                if signed_pred_y != labels[i]:
                    w += labels[i] * self.new_train_x[i]
                    this_num_error = self.num_error(w)
                    if this_num_error < min_num_error:
                        min_num_error = this_num_error
                        best_w = np.copy(w)
                    # print("best_w: " + str(best_w) + " w: " + str(w) + " this_num_error: " + str(
                    #     this_num_error) + " times: " + str(times))
                    break
            times += 1
        self.w = best_w

    def test(self):
        num_right = 0
        n = self.test_x.shape[0]
        for i in range(n):
            if self.sign(np.dot(self.w, self.append1(self.test_x[i]))) == self.test_y[i]:
                num_right += 1
        print("accuracy: " + str(num_right / n))


# if __name__ == "__main__":
#     inputs_x = np.array(
#         [[0.2, 0.7], [0.3, 0.3], [0.4, 0.5], [0.6, 0.5], [0.1, 0.4], [0.4, 0.6], [0.6, 0.2], [0.7, 0.4], [0.8, 0.6],
#          [0.7, 0.5]])
#     labels_y = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])
#     # test_x = np.array([0, 1])
#     w = Pocket(inputs_x, labels_y, 20)
#     # print(sign(np.dot(w, append1(test_x))))

if __name__ == "__main__":
    mu1 = np.array([[-5, 0]])
    mu2 = np.array([[0, 5]])
    Sigma = np.array([[1, 0], [0, 1]])
    R = np.linalg.cholesky(Sigma).T
    s1 = np.random.randn(2000, 2) @ R + mu1
    s2 = np.random.randn(2000, 2) @ R + mu2

    plt.plot(*s1.T, '.', label='+1')
    plt.plot(*s2.T, '+', label='-1')
    plt.axis('scaled')
    plt.legend()

    l1 = np.ones(2000)
    l2 = -l1
    inputs = np.concatenate((s1, s2))
    labels = np.concatenate((l1, l2))
    myPLA = Pocket(inputs, labels, 0.8)

    start_time = datetime.datetime.now()
    myPLA.train()
    end_time = datetime.datetime.now()
    print("运行时间：" + str((end_time - start_time).microseconds) + " us")
    myPLA.test()

    plt.show()
