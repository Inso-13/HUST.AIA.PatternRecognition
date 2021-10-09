import torch.optim as optim
from torch.autograd import Variable
from src.cross_entropy import *
from src.softmax import *

train_x = torch.tensor([[3.0, 0.0], [3.0, 6.0], [0.0, 3.0], [-3.0, 0.0]])
train_y = torch.tensor([[0], [0], [1], [2]])

# Training settings
batch_size = 1
input_size = 2
epochs = 10
num_classes = 3

b = torch.zeros(num_classes)
w = Variable(torch.zeros((input_size, num_classes)))
w.requires_grad_(True)
b.requires_grad_(True)
optimizer = optim.SGD([w, b], lr=1, momentum=0)  # 0.01 0.5

train_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(train_x, train_y),
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


for epoch in range(epochs):
    train(train_loader)
