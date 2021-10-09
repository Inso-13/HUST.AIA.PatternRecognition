import csv
from src.PLA import *


class OVO:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.classes_list = self.__generate_list()
        self.PLA_models = [None for _ in range(len(self.classes_list))]
        self.flower_map = {
            "setosa": 0,
            "versicolor": 1,
            "virginica": 2
        }
        self.data_list_x = None
        self.data_list_y = None
        self.data_list_len = 0

    def train(self, data_path):
        self.data_list_x, self.data_list_y = self.__readcsv(data_path)
        self.data_list_len = len(self.data_list_x)

        for i in range(len(self.classes_list)):
            train_x, train_y, test_x, test_y = self.__generate_dataset(i)
            self.PLA_models[i] = PLA(train_x, train_y, test_x, test_y)
            self.PLA_models[i].train()
            self.PLA_models[i].test()

    def test(self, test_path):
        test_x, test_y = self.__readcsv(test_path)
        test_data_len = len(test_x)
        right = 0
        for i in range(test_data_len):
            votes = []
            for j in range(len(self.classes_list)):
                ret = self.PLA_models[j].test_pos_neg(test_x[i])
                if ret == 1:
                    votes.append(self.classes_list[j][0])
                else:
                    votes.append(self.classes_list[j][1])
            final_vote = max(votes, key=lambda v: votes.count(v))
            if final_vote == test_y[i]:
                right += 1
        print("Test accuracy: {}".format(right / test_data_len))
        return right / test_data_len

    def __generate_list(self):
        n = self.num_classes
        temp_list = []
        for i in range(n):
            for j in range(i + 1, n):
                temp_list.append([i, j])
        return temp_list

    def __readcsv(self, files):
        csvfile = open(files, 'r')
        plots = csv.reader(csvfile, delimiter=',')  # 按行读取CSV文件中的数据,每一行以空格作为分隔符，再将内容保存成列表的形式
        next(plots)  # 读取首行
        x = []
        y = []
        for row in plots:
            x.append([float(row[1]), float(row[2]), float(row[3]), float(row[4])])
            y.append(self.flower_map[row[5]])
        return x, y

    def __generate_dataset(self, index):
        pos, neg = self.classes_list[index]
        count_pos = 0
        count_neg = 0
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for i in range(self.data_list_len):
            item_x = self.data_list_x[i]
            item_y = self.data_list_y[i]
            if item_y == pos:
                if count_pos < 30:
                    count_pos += 1
                    train_x.append(item_x)
                    train_y.append(1)
                else:
                    test_x.append(item_x)
                    test_y.append(1)
            elif item_y == neg:
                if count_neg < 30:
                    count_neg += 1
                    train_x.append(item_x)
                    train_y.append(-1)
                else:
                    test_x.append(item_x)
                    test_y.append(-1)
        return train_x, train_y, test_x, test_y
