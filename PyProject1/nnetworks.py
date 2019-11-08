import numpy as np
import copy
from sklearn.datasets import make_blobs, make_moons
import torch
import time
from torch import nn
import torch.nn.functional as F

# x = torch.randn((60, 3))
# xs = np.array_split(x, 11)
# print(xs)
# print(x.size()[0])
# x = torch.exp(x)
# y = x.sum(dim=1).view(-1, 1).expand_as(x)
# print(x / y)
module = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=2, padding=0)
x = torch.ones(4, 3, 32, 32)
conv1 = nn.Conv2d(3, 10, 5)
pool = nn.MaxPool2d(2, 2)
conv2 = nn.Conv2d(10, 16, 5)
x = pool(F.relu(conv1(x)))
x = pool(F.relu(conv2(x)))
exit(0)

class Module:
    def forward(self, x):
        raise NotImplementedError()

    def backward(self, d):
        raise NotImplementedError()

    def update(self, alpha):
        pass


class Linear(Module):
    def __init__(self, in_features, out_features, device='cpu'):
        self.w = torch.randn((out_features, in_features + 1), device=device) * np.sqrt(
            2 / (in_features + out_features + 1))
        self.outs_grad = None
        self.ins = None

    def forward(self, x):
        if len(x.size()) == 1:
            x = x.view(1, -1)
        ones = torch.ones((x.shape[0], 1), device=self.w.device)
        self.ins = torch.cat((x, ones), dim=1)
        return self.ins.matmul(self.w.t())

    def backward(self, d):
        self.outs_grad = d
        return d.matmul(self.w)[:, :-1]

    def update(self, alpha):
        self.w -= alpha * self.outs_grad.t().matmul(self.ins) / self.ins.size()[0]


class ReLU(Module):
    def __init__(self):
        self.ins = None

    def forward(self, x):
        if len(x.size()) == 1:
            x = x.view(1, -1)
        self.ins = x
        return x * (x > 0)

    def backward(self, d):
        return d * (self.ins > 0)


class Softmax(Module):
    def __init__(self):
        self.ins = None
        self.outs = None

    def forward(self, x):
        if len(x.size()) == 1:
            x = x.view(1, -1)
        self.ins = x
        exps = torch.exp(x)
        sums = exps.sum(dim=1).view(-1, 1).expand_as(x)
        self.outs = exps / sums
        return self.outs

    def backward(self, d):
        x1 = -(d * self.outs).sum(dim=1)
        return self.outs * (x1.view(-1, 1).expand_as(self.outs) + d)


xc = torch.randn((10000, 5000)).cuda()
yc = torch.randn((5000, 10000)).cuda()
x = torch.randn((10000, 5000))
y = torch.randn((5000, 10000))
st = time.time()
xc.matmul(yc)
print(time.time() - st)
st = time.time()
x.matmul(y)
print(time.time() - st)
exit(0)


# y = torch.randn((5, 3)).cuda()
# z = x.matmul(y)
# exit(0)


class MLPClassifier:
    def __init__(self, modules, epochs=3, alpha=0.1, batch_size=2000, device='cpu'):
        self.modules = modules
        self.epochs = epochs
        self.alpha = alpha
        self.batch_size = batch_size
        self.device = device

    def forwarding(self, batch):
        for module in self.modules:
            batch = module.forward(batch)
        return batch

    def backwarding(self, d):
        for module in self.modules[::-1]:
            d = module.backward(d)

    def updating(self):
        for module in self.modules:
            module.update(self.alpha)

    def fit(self, X, y):
        X = torch.tensor(X, device=self.device, dtype=torch.float32)
        y = torch.tensor(y, device=self.device, dtype=torch.float32)
        start = time.time()
        for j in range(self.epochs):
            print(j, end=' ')
            inds = np.arange(X.size()[0])
            np.random.shuffle(inds)
            X = X[inds]
            y = y[inds]

            Xs = torch.split(X, X.size()[0] // self.batch_size + 1)
            ys = torch.split(y, y.size()[0] // self.batch_size + 1)

            for X_b, y_b in zip(Xs, ys):
                outs = self.forwarding(X_b)
                d = torch.zeros(outs.size(), device=self.device)
                for i in range(X_b.size()[0]):
                    d[i][int(y_b[i])] = -1 / outs[i][int(y_b[i])]
                self.backwarding(d)
                self.updating()
        # print('fitting time', time.time() - start)

    def predict_proba(self, X):
        X = torch.tensor(X, device=self.device, dtype=torch.float32)
        return self.forwarding(X)

    def predict(self, X):
        X = torch.tensor(X, device=self.device, dtype=torch.float32)
        p = self.forwarding(X)
        return torch.argmax(p, axis=1)


X, y = make_moons(4000, noise=0.075)
X_test, y_test = make_moons(400, noise=0.075)

start = time.time()
p = MLPClassifier([
    Linear(2, 600, device='cuda'),
    ReLU(),
    Linear(600, 2, device='cuda'),
    Softmax()
], device='cuda')
p.fit(X, y)
pred = np.array(torch.tensor(p.predict(X_test), device='cpu'))
print("Accuracy", np.mean(pred == y_test))
print(time.time() - start)

start = time.time()
p = MLPClassifier([
    Linear(2, 600),
    ReLU(),
    Linear(600, 2),
    Softmax()
])
p.fit(X, y)
pred = np.array(torch.tensor(p.predict(X_test), device='cpu'))
print("Accuracy", np.mean(pred == y_test))
print(time.time() - start)

# X, y = make_blobs(400, 2, centers=[[0, 0], [2.5, 2.5], [-2.5, 3]])
# X_test, y_test = make_blobs(400, 2, centers=[[0, 0], [2.5, 2.5], [-2.5, 3]])
# best_acc = 0
# for _ in range(25):
#     p = MLPClassifierGPU([
#         Linear(2, 64, on_cuda=True),
#         ReLU(),
#         Linear(64, 64),
#         ReLU(),
#         Linear(64, 3),
#         Softmax()
#     ])
#
#     p.fit(X, y)
#     print('fitted')
#     pred = np.array(p.predict(X_test))
#     best_acc = max(np.mean(pred == y_test), best_acc)
# print("Accuracy", best_acc)
