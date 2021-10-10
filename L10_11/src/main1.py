import csv
import torch
from src.BPNet import *

epochs = 1000
lr = 0.05
batch_size = 1

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

train_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(train_x, train_y),
                                           batch_size=batch_size,
                                           shuffle=True, drop_last=True)

test_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(test_x, test_y),
                                          batch_size=batch_size,
                                          shuffle=True, drop_last=True)

net = Net(n_feature=4, n_hidden=20, n_output=3)
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
loss_func = torch.nn.CrossEntropyLoss()

for epoch in range(epochs):
    for batch_idx, (data, label) in enumerate(train_loader):
        optimizer.zero_grad()
        out = net(data)
        loss = loss_func(out, label)
        loss.backward()
        optimizer.step()

correct = 0
for data, target in test_loader:
    out = net(data)
    prediction = torch.max(out, 1)[1]
    pred_y = prediction.data.numpy()
    target_y = target.data.numpy()
    if target_y == pred_y:
        correct += 1
accuracy = correct / len(test_loader)
print("Accuracy: ", accuracy)
