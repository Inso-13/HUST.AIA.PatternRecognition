import torch
import numpy as np


def cross_entropy(y, y_label):
    y_label = make_one_hot(y_label, y.shape[1])
    return torch.sum(-y_label * torch.log(y))


def make_one_hot(input, num_classes):
    input = input.view(-1, 1)
    batch_size = input.shape[0]
    result = torch.zeros((batch_size, num_classes))
    result = result.scatter_(1, input, 1)
    return result
