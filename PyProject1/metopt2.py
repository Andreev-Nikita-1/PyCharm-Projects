import numpy as np
import scipy
import scipy.optimize as opt
import sklearn
import sklearn.datasets
import matplotlib.pyplot as plt
import sys
import time
from scipy.optimize import bracket
from sklearn.model_selection import train_test_split


def sigma(w, x):
    return 1 / (1 + np.exp(-np.dot(w, x)))


oracle_counter = 0


def function(w, X):
    global oracle_counter
    oracle_counter += 1
    return -1 / len(X) * np.sum([np.log(sigma(w, (2 * l - 1) * x)) for x, l in zip(X, labels)])


def gradient(w, X):
    global oracle_counter
    oracle_counter += 1
    return -1 / len(X) * np.sum([x * (l - sigma(w, x)) for x, l in zip(X, labels)], axis=0)


def gessian(w, X):
    return -1 / len(X) * np.sum([np.outer(x, x) * ((sigma(w, x) - 1) * sigma(w, x)) for x in X], axis=0)


def der(fun, point, epsilon=np.sqrt(sys.float_info.epsilon)):
    return (fun(point + epsilon) - fun(point)) / epsilon


def check_gradient(fun, grad, R, dim, args=(), diff_eps=np.sqrt(sys.float_info.epsilon)):
    w = np.random.random(dim)
    w = (2 * w - 1) * R
    dw = np.eye(dim)
    difs = [np.abs((np.dot(grad(w, *args), dw_i) - der(lambda t: fun(w + t * dw_i, *args), 0, diff_eps)) / np.dot(
        grad(w, *args), dw_i)) for dw_i in dw]
    return np.average(difs)


def check_gessian(grad, gess, R, dim, args=(), diff_eps=np.sqrt(sys.float_info.epsilon)):
    w = np.random.random(dim)
    w = (2 * w - 1) * R
    dw = np.eye(dim)
    g = gess(w, *args)

    def xAy(A, x, y):
        return np.dot(x, np.dot(A, y))

    difs = [np.abs((xAy(g, dw1, dw2) - der(lambda t: np.dot(grad(w + t * dw1, *args), dw2), 0, diff_eps))
                   / xAy(g, dw1, dw2)) for dw1 in dw for dw2 in dw]
    return np.average(difs)


def golden_search_bounded(fun, a0, b0, eps=0.0001, args=()):
    ratio = (1 + 5 ** 0.5) / 2

    def step(a, b, c, fc):
        if b - a < eps:
            return a, fun(a, *args)
        else:
            d = a + b - c
            fd = fun(d)
            if c > d:
                c, d = d, c
                fc, fd = fd, fc
            if fc < fd:
                return step(a, d, c, fc)
            else:
                return step(c, b, d, fd)

    c0 = a0 + (b0 - a0) / ratio
    return step(a0, b0, c0, fun(c0, *args))[0]


def golden_search(fun, a=0, b=1, eps=0.0001, args=()):
    x = golden_search_bounded(fun, a, b, eps, args)
    d = b - a
    if np.abs(x - a) < eps:
        return golden_search(fun, a - 10 * d, a, eps, args)
    if np.abs(x - b) < eps:
        return golden_search(fun, b, b + 10 * d, eps, args)
    return x


def armiho(fun, c=0.5, x0=1, df0=None):
    x = x0
    f0 = fun(0)
    if df0 is None:
        df0 = der(fun, 0)
    while fun(x) > f0 - c * x * df0:
        x /= 2
    return x


def nester(fun, f0, d, L0=1):
    L = L0
    while fun(1 / L) > f0 - 1 / (2 * L) * np.dot(d, d):
        L *= 2
    return L


def gradient_descent(fun, grad, start, one_dim_search='brent', args=(), search_kwargs=dict([]), epsilon=0.0001,
                     true_min=0):
    iterations, oracle, times, accuracies, grad_ratios = [], [], [], [], []
    start_time, k = time.time(), 0
    x = start
    d0 = grad(x, *args)
    d = d0
    ratio = 1
    if one_dim_search == 'nester':
        L, L0 = 2, 0
        if 'L0' in search_kwargs.keys():
            L0 = 2 * search_kwargs['L0']
            L = L0

    while ratio > epsilon:
        iterations.append(k)
        k += 1
        oracle.append(oracle_counter)
        times.append(time.time() - start_time)
        f0 = fun(x, *args)
        accuracies.append(f0 - true_min)
        grad_ratios.append(ratio)
        f = lambda alpha: fun(x - d * alpha, *args)

        if one_dim_search == 'golden':
            alpha = golden_search(f, **search_kwargs)
        elif one_dim_search == 'brent':
            alpha = opt.minimize_scalar(f)['x']
        elif one_dim_search == 'armiho':
            alpha = armiho(f, **search_kwargs, df0=-np.dot(d, d))
        elif one_dim_search == 'wolf':
            alpha = opt.line_search(fun, grad, x, -d, **search_kwargs, gfk=d, args=args)[0]
        elif one_dim_search == 'nester':
            L = nester(f, f0, d, L0=max(L / 2, L0))
            alpha = 1 / L

        x = x - d * alpha
        d = grad(x, *args)
        ratio = np.dot(d, d) / np.dot(d0, d0)

    return x, iterations, oracle, times, accuracies, grad_ratios


def polinome(x):
    return x ** 2


def normalize(array):
    mins = np.array([min(array[:, k]) for k in range(len(array[0]))])
    maxs = np.array([max(array[:, k]) for k in range(len(array[0]))])
    array = np.array([(vector - mins) / (maxs - mins) for vector in array])
    return array


def graph(x, y, x_l=None, y_l=None, title=None):
    fig, ax = plt.subplots()
    ax.plot(x, [np.log(t) for t in y])
    ax.set(xlabel=x_l, ylabel=y_l,
           title=title)
    ax.grid()
    plt.show()


cancer = sklearn.datasets.load_breast_cancer()
# X = np.random.random((100, 3))
# labels = np.random.randint(0, 2, (100))
X = normalize(cancer['data'])
labels = cancer['target']
X, X_test, labels, labels_test = train_test_split(X, labels, test_size=0.2, shuffle=False)
print(check_gradient(function, gradient, 1, X.shape[1], args=[X]))
print(check_gessian(gradient, gessian, 1, X.shape[1], args=[X]))

w0 = np.array([0.52658227, 0.37548235, 0.27720264, 0.82988368, 0.63424511, 0.34688084
                  , 0.05520993, 0.15297424, 0.00745145, 0.08471717, 0.5714892, 0.62753455
                  , 0.43193636, 0.28142003, 0.91129921, 0.75423357, 0.93720731, 0.73454387
                  , 0.11892809, 0.44075745, 0.44719281, 0.65085188, 0.09238662, 0.46245895
                  , 0.20133476, 0.82547662, 0.843942, 0.87022425, 0.1425544, 0.16122998]
              )
# w0 = np.random.random(30)
w_true = np.array([302.5925661, -21.69111231, 207.55842006, -411.28849642, -32.24866798
                      , -4.31827468, -34.71080755, -73.71085179, 38.02308458, 49.91720216
                      , 66.57753268, 15.52855102, 27.0831392, -312.8996407, 24.89967549
                      , -34.07416117, 155.51644952, -106.31990291, 29.20772399, 65.4730172
                      , 82.76286878, -19.74667188, 15.59046806, -451.20587395, 13.58897497
                      , 79.36381994, -51.81633846, 19.5727145, -65.71161459, -102.40926345])
print(np.mean(np.array([round(sigma(w_true, x)) for x in X_test]) == labels_test))
start = time.time()
w_res, iterations, oracle, times, accuracy, grad_ratio = gradient_descent(function,
                                                                          gradient, w0, args=[X],
                                                                          one_dim_search='nester',
                                                                          # search_kwargs=dict(
                                                                          #     [('c', 0.5), ('x0', 10)]),
                                                                          epsilon=0.000001,
                                                                          true_min=function(w_true, X))
print(np.linalg.norm(w_res - w_true))
print(np.mean(np.array([round(sigma(w_res, x)) for x in X_test]) == labels_test))
print(time.time() - start)
# graph(iterations[5:], accuracy[5:], title='1.1')
# graph(times[5:], accuracy[5:], title='2.1')
# graph(oracle[5:], accuracy[5:], title='3.1')
# graph(iterations[5:], grad_ratio[5:], title='1.2')
# graph(times[5:], grad_ratio[5:], title='2.2')
# graph(oracle[5:], grad_ratio[5:], title='3.2')

# print(*golden_search(polinome, -2, 2))
# minimization = opt.minimize_scalar(polinome)
# print(minimization['x'], minimization['fun'])
