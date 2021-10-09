import numpy as np
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from src.cross_entropy import *
from src.softmax import *
import matplotlib.pyplot as plt

# Training settings
batch_size = 256
input_size = 28 * 28
epochs = 10
num_classes = 10

# MNIST Dataset
train_dataset = datasets.MNIST(root='../data/',
                               train=True,
                               transform=transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()]),
                               download=True)

test_dataset = datasets.MNIST(root='../data/',
                              train=False,
                              transform=transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()]))

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True, drop_last=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True, drop_last=True)

w = torch.zeros((input_size, num_classes))
b = torch.zeros(num_classes)
w = Variable(torch.normal(w, 0.01))
w.requires_grad_(True)
b.requires_grad_(True)
optimizer = optim.SGD([w, b], lr=0.001, momentum=0.1)  # 0.01 0.5


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
plt.subplot(3, 5, 2)
plt.title("losses")
plt.plot(losses)
plt.subplot(3, 5, 3)
plt.title("train_accuracy")
plt.plot(train_accuracy)
plt.subplot(3, 5, 4)
plt.title("test_accuracy")
plt.plot(test_accuracy)
with torch.no_grad():
    for i in range(10):
        data, target = test_dataset.__getitem__(i)
        pred = np.argmax(softmax(torch.add(torch.matmul(w.T, data.view(-1)).T, b)))
        plt.subplot(3, 5, i + 6)
        plt.title("{}->{}".format(target, pred))
        plt.imshow(data.view(28, 28))
plt.show()
