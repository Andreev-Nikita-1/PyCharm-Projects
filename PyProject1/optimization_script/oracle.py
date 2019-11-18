import numpy as np


def randd():
    print(np.random.random())


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def oracle(w, X, labels, order=0, no_function=False):
    Xw = X.dot(w)
    sigmoids = sigmoid(labels * Xw)

    f = 0
    if not no_function:
        f = -1 / X.shape[0] * np.sum(np.log(sigmoids))

    if order == 0:
        return f

    grad_coeffs = labels * (1 - sigmoids)
    X1 = X.multiply(grad_coeffs.reshape(-1, 1))
    g = -1 / X.shape[0] * np.array(X1.sum(axis=0)).reshape(X.shape[1])

    if order == 1:
        return f, g, 0

    hess_coeffs = sigmoids * (1 - sigmoids)
    h = lambda v: 1 / X.shape[0] * X.transpose().dot(X.dot(v) * hess_coeffs)

    if order == 2:
        return f, g, h
