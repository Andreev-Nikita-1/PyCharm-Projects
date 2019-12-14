import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Oracle:
    def __init__(self, X, labels, stochastic=False, Q_reccurent=None, batch_size=None):
        self.X = X
        self.labels = labels
        self.fc = 0
        self.gc = 0
        self.hc = 0
        self.stochastic = stochastic

        if stochastic:
            self.batch_size = batch_size
            self.X_batches = None
            self.labels_batches = None
            self.batch_index = None
            self.reload()
            self.Q = 0
            self.Q_reccurent = Q_reccurent

    def get_start(self):
        return np.random.randn(self.X.shape[1])

    def reload(self):
        inds = np.arange(self.X.shape[0])
        np.random.shuffle(inds)
        self.X_batches = np.split(self.X[inds], self.X.shape[0] // self.batch_size + 1)
        self.labels_batches = np.split(self.labels[inds], self.X.shape[0] // self.batch_size + 1)
        self.batch_index = -1

    def batch(self):
        if self.batch_index >= len(self.X_batches) - 1:
            self.reload()
        self.batch_index += 1
        return self.X_batches[self.batch_index], self.labels_batches[self.batch_index]

    def evaluate(self, w, order=0, no_function=False):
        if self.stochastic:
            X, labels = self.batch()
        else:
            X = self.X
            labels = self.labels

        Xw = X.dot(w)
        sigmoids = sigmoid(labels * Xw)

        f = 0
        if not no_function:
            f = -1 / X.shape[0] * np.sum(np.log(sigmoids))
            self.fc += 1

        if order == 0:
            return f

        grad_coeffs = labels * (1 - sigmoids)
        X1 = X.multiply(grad_coeffs.reshape(-1, 1))
        g = -1 / X.shape[0] * np.array(X1.sum(axis=0)).reshape(X.shape[1])
        self.gc += 1

        if order == 1:
            return f, g, 0

        hess_coeffs = sigmoids * (1 - sigmoids)

        def hessian_mult(x):
            self.hc += 1
            return 1 / X.shape[0] * X.transpose().dot(X.dot(x) * hess_coeffs)

        h = hessian_mult

        if order == 2:
            return f, g, h
