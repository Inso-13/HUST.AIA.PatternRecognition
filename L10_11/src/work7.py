import torch
import torchvision as tv
import matplotlib.pyplot as plt
import numpy as np

epochs = 1
batch_size = 1
lr = 0.01


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)
        self.out = torch.nn.Linear(n_hidden2, n_output)
        torch.nn.init.constant(self.hidden1.bias, 1)
        torch.nn.init.constant(self.hidden1.weight, 1)
        torch.nn.init.constant(self.hidden2.bias, 1)
        torch.nn.init.constant(self.hidden2.weight, 1)
        torch.nn.init.constant(self.out.bias, 1)
        torch.nn.init.constant(self.out.weight, 1)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.out(x)
        return x


train_x = torch.tensor([[1.0, 1.0], [-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0]])
train_y = torch.tensor([[1.0], [1.0], [-1.0], [-1.0]])
train_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(train_x, train_y),
                                           batch_size=batch_size,
                                           shuffle=True, drop_last=True)

net = Net(2, 2, 3, 1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr)


def train(train_dataloader):
    for batch_idx, (data, target) in enumerate(train_dataloader):
        optimizer.zero_grad()
        output = net(data)
        # loss
        loss = criterion(output, target)
        loss.backward()
        # update
        optimizer.step()


for epoch in range(epochs):
    train(train_loader)
    print("ok")
