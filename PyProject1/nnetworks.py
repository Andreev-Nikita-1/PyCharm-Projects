import numpy as np
import copy
from sklearn.datasets import make_blobs, make_moons
import torch


# x = torch.randn((60, 3))
# xs = np.array_split(x, 11)
# print(xs)
# print(x.size()[0])
# x = torch.exp(x)
# y = x.sum(dim=1).view(-1, 1).expand_as(x)
# print(x / y)


class Module:
    def forward(self, x):
        raise NotImplementedError()

    def backward(self, d):
        raise NotImplementedError()

    def update(self, alpha):
        pass


class Linear(Module):
    def __init__(self, in_features, out_features):
        self.w = torch.randn((out_features, in_features + 1)) * np.sqrt(2 / (in_features + out_features + 1))
        self.outs_grad = None
        self.ins = None

    def forward(self, x):
        if len(x.size()) == 1:
            x = x.view(1, -1)
        self.ins = torch.cat((x, torch.ones(x.shape[0], 1)), dim=1)
        return self.ins.matmul(self.w.t())

    def backward(self, d):
        self.outs_grad = d
        return d.matmul(self.w)[:, :-1]

    def update(self, alpha):
        self.w -= alpha * self.outs_grad.t().matmul(self.ins)


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


class MLPClassifier:
    def __init__(self, modules, epochs=20, alpha=0.01, batch_size=20):
        self.modules = modules
        self.epochs = epochs
        self.alpha = alpha
        self.batch_size = batch_size

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
        X = torch.Tensor(X)
        y = torch.Tensor(y)
        for i in range(self.epochs):
            # outs = self.forwarding(X)
            # errs = ([-np.log(outs[i][int(y[i])]) for i in range(X.size()[0])])
            # print('-', torch.Tensor(errs).sum(dim=0))

            inds = np.arange(X.size()[0])
            np.random.shuffle(inds)
            X = X[inds]
            y = y[inds]

            Xs = np.array_split(X, X.size()[0] // self.batch_size + 1)
            ys = np.array_split(y, y.size()[0] // self.batch_size + 1)

            for X_b, y_b in zip(Xs, ys):
                outs = self.forwarding(X_b)
            d = torch.zeros(outs.size())
            for i in range(X_b.size()[0]):
                d[i][int(y_b[i])] = -1 / outs[i][int(y_b[i])]
            self.backwarding(d)
            self.updating()

    def predict_proba(self, X):
        X = torch.Tensor(X)
        return self.forwarding(X)

    def predict(self, X):
        X = torch.Tensor(X)
        p = self.predict_proba(X)
        return np.argmax(p, axis=1)


X, y = make_moons(400, noise=0.075)
X_test, y_test = make_moons(400, noise=0.075)

best_acc = 0
for _ in range(25):
    p = MLPClassifier([
        Linear(2, 64),
        ReLU(),
        Linear(64, 64),
        ReLU(),
        Linear(64, 2),
        Softmax()
    ])

    p.fit(X, y)
    pred = np.array(p.predict(X_test))
    best_acc = max(np.mean(pred == y_test), best_acc)
print("Accuracy", best_acc)
