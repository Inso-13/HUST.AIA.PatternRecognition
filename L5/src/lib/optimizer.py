import numpy as np
import random


class base_optimizer:
    def __init__(self, params, grad_fn):
        self.params = params
        self.grad_fn = grad_fn

    def __str__(self):
        return "base_optimizer"

    def step(self, x):
        self.params = self.params


class GD(base_optimizer):
    def __init__(self, params, grad_fn, lr=0.4):
        super().__init__(params, grad_fn)
        self.lr = lr

    def __str__(self):
        return "GD"

    def step(self):
        self.params = self.params - self.lr * self.grad_fn(self.params)


class SGD(GD):
    def __init__(self, params, grad_fn, X, Y, lr=0.4, batch_size=1):
        super().__init__(params, grad_fn)
        self.X = X
        self.Y = Y
        self.lr = lr
        self.batch_size = batch_size
        self.lenX = X.shape[0]

    def step(self):
        grad = 0
        for i in range(self.batch_size):
            j = random.randint(0, self.lenX-self.batch_size)
            grad += self.grad_fn(self.params, self.X[j], self.Y[j])
        self.params = self.params - self.lr * grad / self.batch_size
        return self.params

    def __str__(self):
        return "SGD"


class Momentum(base_optimizer):
    def __init__(self, params, grad_fn, lr=0.4, lam=0.9):
        super().__init__(params, grad_fn)
        self.lr = lr
        self.lam = lam
        self.mt = 0

    def __str__(self):
        return "Momentum"

    def step(self):
        self.mt = self.lam * self.mt - self.lr * self.grad_fn(self.params)
        self.params = self.params + self.mt


class Adagrad(base_optimizer):
    def __init__(self, params, grad_fn, lr=0.4, eps=1e-6):
        super().__init__(params, grad_fn)
        self.lr = lr
        self.eps = eps
        self.grads = []

    def __str__(self):
        return "Adagrad"

    def step(self):
        grad = self.grad_fn(self.params)
        self.grads.append(grad)
        sigma = np.sqrt(self.eps + np.sum(np.array(self.grads)
                                          ** 2) / (len(self.grads) + 1))
        self.params = self.params - self.lr * self.grad_fn(self.params) / sigma


class RMSProp(base_optimizer):
    def __init__(self, params, grad_fn, lr=0.4, eps=1e-6, alpha=0.9):
        super().__init__(params, grad_fn)
        self.lr = lr
        self.eps = eps
        self.alpha = 0.9
        self.last_sigma = self.eps

    def __str__(self):
        return "RMSProp"

    def step(self):
        grad = self.grad_fn(self.params)
        sigma = np.sqrt(self.alpha * self.last_sigma **
                        2 + (1 - self.alpha) * grad ** 2)
        self.params = self.params - self.lr * grad / sigma
        self.last_sigma = sigma


class Adam(base_optimizer):
    def __init__(self, params, grad_fn, lr=0.4, eps=1e-6, beta1=0.99, beta2=0.999):
        super().__init__(params, grad_fn)
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr = lr
        self.eps = eps
        self.mt = 0
        self.nt = 0
        self.t = 0

    def __str__(self):
        return "Adam"

    def step(self):
        self.t += 1
        grad = self.grad_fn(self.params)
        self.mt = self.beta1 * self.mt + (1 - self.beta1) * grad
        self.nt = self.beta2 * self.nt + (1 - self.beta2) * grad ** 2
        mt_b = self.mt / (1 - self.beta1 ** self.t)
        nt_b = self.nt / (1 - self.beta2 ** self.t)
        self.params = self.params - mt_b / (nt_b ** 0.5 + self.eps)
