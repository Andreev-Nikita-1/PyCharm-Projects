import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.optimize as opt
import sklearn.datasets
from sklearn.model_selection import train_test_split


def sigma(w, x):
    return 1 / (1 + np.exp(-np.dot(w, x)))


def function(w, X):
    return -1 / len(X) * np.sum([np.log(sigma(w, (2 * l - 1) * x)) for x, l in zip(X, labels)])


def gradient(w, X):
    return -1 / len(X) * np.sum([x * (l - sigma(w, x)) for x, l in zip(X, labels)], axis=0)


def hessian(w, X):
    sigmas = np.array([sigma(w, x) for x in X])
    return -1 / len(X) * np.sum([np.outer(x, x) * ((sigmas[i] - 1) * sigmas[i]) for i, x in enumerate(X)], axis=0)


def der(fun, point, epsilon=np.sqrt(sys.float_info.epsilon)):
    return (fun(point + epsilon) - fun(point)) / epsilon


def check_gradient(fun, grad, R, dim, args=(), diff_eps=np.sqrt(sys.float_info.epsilon)):
    w = np.random.random(dim)
    w = (2 * w - 1) * R
    dw = np.eye(dim)
    difs = [np.abs((np.dot(grad(w, *args), dw_i) - der(lambda t: fun(w + t * dw_i, *args), 0, diff_eps)) / np.dot(
        grad(w, *args), dw_i)) for dw_i in dw]
    return np.average(difs)


def check_hessian(grad, hess, R, dim, args=(), diff_eps=np.sqrt(sys.float_info.epsilon)):
    w = np.random.random(dim)
    w = (2 * w - 1) * R
    dw = np.eye(dim)
    g = hess(w, *args)

    def xAy(A, x, y):
        return np.dot(x, np.dot(A, y))

    difs = [np.abs((xAy(g, dw1, dw2) - der(lambda t: np.dot(grad(w + t * dw1, *args), dw2), 0, diff_eps))
                   / xAy(g, dw1, dw2)) for dw1 in dw for dw2 in dw]
    return np.average(difs)


def golden_search_bounded(fun, a0, b0, eps=0.0001, args=()):
    ratio = (1 + 5 ** 0.5) / 2

    def step(a, b, c, fc, onumber):
        if b - a < eps:
            return a, fun(a, *args), onumber + 1
        else:
            d = a + b - c
            fd = fun(d)
            if c > d:
                c, d = d, c
                fc, fd = fd, fc
            if fc < fd:
                return step(a, d, c, fc, onumber + 1)
            else:
                return step(c, b, d, fd, onumber + 1)

    c0 = a0 + (b0 - a0) / ratio
    solution = step(a0, b0, c0, fun(c0, *args), 0)
    return solution[0], solution[2]


def golden_search(fun, eps=0.0001, args=()):
    a, _, b, _, _, _, onumber = opt.bracket(fun, args=args)
    if b < a:
        a, b = b, a
    gsb = golden_search_bounded(fun, a, b, eps=eps, args=args)
    return gsb[0], gsb[1] + onumber


def armiho(fun, c=0.5, x0=1, df0=None):
    x = x0
    f0 = fun(0)
    oracle = 1

    if df0 is None:
        df0 = der(fun, 0)
    while fun(x) > f0 + c * x * df0:
        x /= 2
        oracle += 1
    oracle += 1

    return x, oracle


def nester(fun, f0, d, L0=1):
    L = L0
    oracle = 0

    while fun(1 / L) > f0 - 1 / (2 * L) * np.dot(d, d):
        L *= 2
        oracle += 1
    oracle += 1

    return L, oracle


def solve(G, d):
    L = np.linalg.cholesky(G)
    Ltx = scipy.linalg.solve_triangular(L, d, lower=True)
    return scipy.linalg.solve_triangular(np.transpose(L), Ltx, lower=False)


def optimization_task(fun, grad, start, method='gradient descent', hess=None, one_dim_search=None, args=(),
                      search_kwargs=dict([]), epsilon=0.0001, true_min=0):
    iterations, oracles, times, accuracies, grad_ratios = [], [], [], [], []
    start_time, k, oracle = time.time(), 0, 0
    x = start
    g0 = grad(x, *args)
    oracle += 1

    if one_dim_search is None and method == 'gradient descent':
        one_dim_search = 'brent'
    elif one_dim_search is None and method == 'newton':
        one_dim_search = 'unit step'

    if one_dim_search == 'nester':
        L, L0 = 2, 0
        if 'L0' in search_kwargs.keys():
            L0 = 2 * search_kwargs['L0']
            L = L0

    while True:
        gk = grad(x, *args)
        fk = fun(x, *args)
        if method == 'gradient descent':
            d = -gk
        elif method == 'newton':
            d = -solve(hess(x, *args), gk)
        oracle += 1
        ratio = np.dot(gk, gk) / np.dot(g0, g0)
        iterations.append(k)
        k += 1
        oracles.append(oracle)
        times.append(time.time() - start_time)
        accuracies.append(fk - true_min)
        grad_ratios.append(ratio)

        if ratio <= epsilon:
            break

        f = lambda alpha: fun(x + d * alpha, *args)

        if one_dim_search == 'unit step':
            alpha = 1
        elif one_dim_search == 'golden':
            alpha, oracle_counter = golden_search(f, **search_kwargs)
            oracle += oracle_counter
        elif one_dim_search == 'brent':
            solution = opt.minimize_scalar(f)
            alpha = solution['x']
            oracle += solution['nfev']
        elif one_dim_search == 'armiho':
            alpha, oracle_counter = armiho(f, **search_kwargs, df0=np.dot(gk, d))
            oracle += oracle_counter
        elif one_dim_search == 'wolf':
            solution = opt.line_search(fun, grad, x, d, **search_kwargs, gfk=gk, old_fval=fk, args=args)
            alpha = solution[0]
            oracle += solution[1]
        elif one_dim_search == 'nester':
            L, oracle_counter = nester(f, fk, gk, L0=max(L / 2, L0))
            alpha = 1 / L
            oracle += oracle_counter

        x = x + d * alpha

    return x, iterations, oracles, times, accuracies, grad_ratios


def polinome(x):
    return x ** 2 / 2


def normalize(array):
    def add1(v):
        l = v.tolist()
        l.append(1)
        return np.array(l)

    mins = np.array([min(array[:, k]) for k in range(len(array[0]))])
    maxs = np.array([max(array[:, k]) for k in range(len(array[0]))])
    array = np.array([(vector - mins) / (maxs - mins) for vector in array])
    array = np.array([add1(v) for v in array])
    return array


def graph(x, y, x_l=None, y_l=None, title=None):
    fig, ax = plt.subplots()
    ax.plot(x, [np.log(t) for t in y])
    ax.set(xlabel=x_l, ylabel=y_l, title=title)
    ax.grid()
    plt.show()


def info(w_res, iterations, oracle, times, accuracies, grad_ratios, w_true):
    print('|x_k - x*| =', np.linalg.norm(w_true - w_res))
    print('iterations =', iterations[-1])
    print('oracle =', oracle[-1])
    print('time =', times[-1])
    print('accuracy =', accuracies[-1])
    print('grad_ratio =', grad_ratios[-1])


cancer = sklearn.datasets.load_breast_cancer()
# X = np.random.random((100, 3))
# labels = np.random.randint(0, 2, (100))
X = normalize(cancer['data'])
labels = cancer['target']
X, X_test, labels, labels_test = train_test_split(X, labels, test_size=0.2)
# print(check_gradient(function, gradient, 1, X.shape[1], args=[X]))
# print(check_hessian(gradient, hessian, 1, X.shape[1], args=[X]))

# w0 = 0.5*np.array([0.52658227, 0.37548235, 0.27720264, 0.82988368, 0.63424511, 0.34688084
#                         , 0.05520993, 0.15297424, 0.00745145, 0.08471717, 0.5714892, 0.62753455
#                         , 0.43193636, 0.28142003, 0.91129921, 0.75423357, 0.93720731, 0.73454387
#                         , 0.11892809, 0.44075745, 0.44719281, 0.65085188, 0.09238662, 0.46245895
#                         , 0.20133476, 0.82547662, 0.843942, 0.87022425, 0.1425544, 0.16122998, 0]
#                     )
w0 = 0.5 * np.random.random(31)

w_true = opt.minimize(function, w0, args=X, jac=gradient)['x']
# w_true = np.array([302.5925661, -21.69111231, 207.55842006, -411.28849642, -32.24866798
#                       , -4.31827468, -34.71080755, -73.71085179, 38.02308458, 49.91720216
#                       , 66.57753268, 15.52855102, 27.0831392, -312.8996407, 24.89967549
#                       , -34.07416117, 155.51644952, -106.31990291, 29.20772399, 65.4730172
#                       , 82.76286878, -19.74667188, 15.59046806, -451.20587395, 13.58897497
#                       , 79.36381994, -51.81633846, 19.5727145, -65.71161459, -102.40926345])
print(np.mean(np.array([round(sigma(w_true, x)) for x in X_test]) == labels_test))
start = time.time()
w_res, iterations, oracles, times, accuracies, grad_ratios = optimization_task(function,
                                                                               gradient, w0,
                                                                               method='gradient descent',
                                                                               hess=hessian,
                                                                               args=[X],
                                                                               one_dim_search='golden',
                                                                               # search_kwargs=dict(
                                                                               # [('c', 0.5), ('x0', 10)]),
                                                                               # [('maxiter', 10)]),
                                                                               epsilon=0.00001,
                                                                               true_min=function(w_true, X))

print(np.mean(np.array([round(sigma(w_res, x)) for x in X_test]) == labels_test))
print(np.linalg.norm(w_res))
info(w_res, iterations, oracles, times, accuracies, grad_ratios, w_true)
# graph(iterations[5:], accuracy[5:], title='1.1')
# graph(times[5:], accuracy[5:], title='2.1')
# graph(oracle[5:], accuracy[5:], title='3.1')
# graph(iterations[5:], grad_ratio[5:], title='1.2')
# graph(times[5:], grad_ratio[5:], title='2.2')
# graph(oracle[5:], grad_ratio[5:], title='3.2')

# print(*golden_search(polinome, -2, 2))
# minimization = opt.minimize_scalar(polinome)
# print(minimization['x'], minimization['fun'])
