import torch
import numpy as np


def softmax(z):
    if len(z.shape) == 2:
        batch_size = z.shape[0]
        num_classes = z.shape[1]
        z_max = np.max(z.detach().numpy(), axis=1)
        z_max = torch.from_numpy(z_max).expand((num_classes, batch_size)).T
        z = torch.subtract(z, z_max)
        exp_z = torch.exp(z)
        sum_z = torch.sum(exp_z, axis=1)
        return torch.div(exp_z.T, sum_z).T
    else:
        z_max = np.max(z.detach().numpy())
        z_max = z_max * torch.ones_like(z)
        z = torch.subtract(z, z_max)
        exp_z = torch.exp(z)
        sum_z = torch.sum(exp_z)
        return torch.div(exp_z.T, sum_z)