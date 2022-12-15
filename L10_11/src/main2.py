import torch
import torchvision as tv
import matplotlib.pyplot as plt
import numpy as np
from lib.LeNet import *

epochs = 10
batch_size = 256
lr = 0.01
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainset = tv.datasets.MNIST(
    root='../data/',
    train=True,
    download=True,
    transform=tv.transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True,
)
testset = tv.datasets.MNIST(
    root='../data/',
    train=False,
    download=True,
    transform=tv.transforms.ToTensor())
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=batch_size,
    shuffle=False,
)

net = LeNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)


def test(data_loader):
    with torch.no_grad():
        correct = 0
        total = 0

        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            predicted = np.argmax(outputs.cpu().detach().numpy(), axis=1)
            total += labels.size(0)
            correct += (torch.tensor(predicted) == labels.cpu()).sum()
        return correct.item() / total


if __name__ == "__main__":
    losses = []
    train_accuracy = []
    test_accuracy = []
    for epoch in range(epochs):
        sum_loss = 0.0

        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()

        losses.append(sum_loss)
        train_accuracy.append(test(trainloader))
        test_accuracy.append(test(testloader))

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

    evalloader = torch.utils.data.DataLoader(
        testset,
        batch_size=1,
        shuffle=True,
    )
    with torch.no_grad():
        i = 0
        for images, labels in evalloader:
            images, labels = images.to(device), labels.to(device)
            i += 1
            if i > 10:
                break
            pred = net(images)
            pred = np.argmax(pred.cpu().detach().numpy())
            plt.subplot(3, 5, i + 5)
            plt.title("{}->{}".format(labels.cpu().item(), pred))
            plt.imshow(images.cpu().view(28, 28))
    plt.show()
