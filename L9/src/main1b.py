import torch.optim as optim
from torch.autograd import Variable
from lib.cross_entropy import *
from lib.softmax import *
import matplotlib.pyplot as plt
import csv

flower_map = {
    "setosa": 0,
    "versicolor": 1,
    "virginica": 2
}


def readcsv(files):
    csvfile = open(files, 'r')
    plots = csv.reader(csvfile, delimiter=',')  # 按行读取CSV文件中的数据,每一行以空格作为分隔符，再将内容保存成列表的形式
    next(plots)  # 读取首行
    x = []
    y = []
    for row in plots:
        x.append([float(row[1]), float(row[2]), float(row[3]), float(row[4])])
        y.append(flower_map[row[5]])
    return torch.tensor(x), torch.tensor(y)


train_x, train_y = readcsv("../data/Iris/iris_train.csv")
test_x, test_y = readcsv("../data/Iris/iris_test.csv")

# Training settings
batch_size = 1
input_size = 4
epochs = 100
num_classes = 3

w = torch.zeros((input_size, num_classes))
b = torch.zeros(num_classes)
w = Variable(torch.normal(w, 0.01))
w.requires_grad_(True)
b.requires_grad_(True)
optimizer = optim.SGD([w, b], lr=0.0008, momentum=0.0)  # 0.01 0.5

train_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(train_x, train_y),
                                           batch_size=batch_size,
                                           shuffle=True, drop_last=True)

test_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(test_x, test_y),
                                          batch_size=batch_size,
                                          shuffle=True, drop_last=True)


def train(train_dataloader):
    avg_loss = 0
    for batch_idx, (data, target) in enumerate(train_dataloader):
        data = data.view(batch_size, -1)
        optimizer.zero_grad()
        output = softmax(torch.add(torch.matmul(w.T, data.view(batch_size, -1).T).T, b))
        # loss
        loss = cross_entropy(output, target)
        avg_loss += loss
        loss.backward()
        # update
        optimizer.step()
    return avg_loss / len(train_dataloader)


def test(test_dataloader):
    correct = 0
    # 测试集
    for data, target in test_dataloader:
        data, target = Variable(data), Variable(target)
        data = data.view(-1)
        output = softmax(torch.add(torch.matmul(w.T, data.view(batch_size, -1).T).T, b))
        pred = np.argmax(output.detach().numpy(), axis=1)
        for i in range(batch_size):
            if pred[i] == target[i]:
                correct += 1
    return correct / len(test_dataloader.dataset)


losses = []
train_accuracy = []
test_accuracy = []
for epoch in range(epochs):
    losses.append(train(train_loader).item())
    train_accuracy.append(test(train_loader))
    test_accuracy.append(test(test_loader))
    print("epoch:{} loss:{} train_accuracy:{} test_accuracy:{}".format(epoch, losses[-1], train_accuracy[-1],
                                                                       test_accuracy[-1]))

plt.xlabel("epochs")
plt.subplot(131)
plt.title("losses")
plt.plot(losses)
plt.subplot(132)
plt.title("train_accuracy")
plt.plot(train_accuracy)
plt.subplot(133)
plt.title("test_accuracy")
plt.plot(test_accuracy)
plt.show()
