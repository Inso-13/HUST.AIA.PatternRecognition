from lib.optimizer import *
import matplotlib.pyplot as plt
import math

if __name__ == "__main__":
    fn = lambda x: x * math.cos(0.25 * math.pi * x)
    grad_fn = lambda x: math.cos(0.25 * math.pi * x) - 0.25 * math.pi * x * math.sin(0.25 * math.pi * x)
    x0 = -4
    epochs = 50
    lr = 0.4

    optimizers = [GD, SGD, Momentum, Adagrad, RMSProp, Adam]
    optimizer_list = [Op(x0, grad_fn, lr=lr) for Op in optimizers]
    sub_num = 0
    plt.xlabel("epoch")
    for optimizer in optimizer_list:
        x = [x0]
        fx = [fn(x0)]
        for epoch in range(epochs):
            optimizer.step()
            x.append(optimizer.params)
            fx.append(fn(optimizer.params))
        sub_num += 1
        plt.subplot(3, 4, sub_num)
        plt.ylabel("x")
        plt.title(optimizer.__str__())
        plt.plot(x)
        sub_num += 1
        plt.subplot(3, 4, sub_num)
        plt.ylabel("f(x)")
        plt.title(optimizer.__str__())
        plt.plot(fx)

    plt.show()
    for optimizer in optimizer_list:
        print("class name: {}, x: {}, f(x): {}".format(optimizer.__class__, optimizer.params, fn(optimizer.params)))
