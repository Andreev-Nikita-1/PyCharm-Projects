import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.optimize as opt
import scipy.sparse
import sklearn.datasets
from sklearn.model_selection import train_test_split

# def sigma(w, x):
#     return 1 / (1 + np.exp(-np.dot(w, x)))

# grad_time = 0

# def function(w, X, labels):
#     return -1 / len(X) * np.sum([np.log(sigma(w, l * x)) for x, l in zip(X, labels)])
#
# def gradient(w, X, labels):
#     global grad_time
#     start = time.time()
#     sum = -1 / len(X) * np.sum([l * x * sigma(w, -l * x) for x, l in zip(X, labels)], axis=0)
#     grad_time += (time.time() - start)
#     return sum

sigmoids = []
arguments = []


def sigmoid(x):
    global sigmoids
    global arguments
    arguments.append(x)
    ans = 1 / (1 + np.exp(-x))
    sigmoids.append(ans)
    return ans


def function(w, X, labels):
    Xw = X.dot(w)
    return -1 / X.shape[0] * np.sum([np.log(sigmoid(l * xw)) for xw, l in zip(Xw, labels)])


def gradient(w, X, labels):
    Xw = X.dot(w)
    coeffs = np.array([l * sigmoid(-l * xw) for xw, l in zip(Xw, labels)])
    if isinstance(X, scipy.sparse.csr_matrix):
        X1 = X.multiply(coeffs.reshape(-1, 1))
    else:
        X1 = coeffs.reshape(-1, 1) * X
    return -1 / X.shape[0] * np.array(X1.sum(axis=0)).reshape(X.shape[1])


def hessian(w, X, labels):
    Xw = X.dot(w)
    coeffs = np.array([sigmoid(xw) * sigmoid(-xw) for xw in Xw])
    return 1 / X.shape[0] * np.sum([np.outer(x, x) * coeffs[i] for i, x in enumerate(X.todense())], axis=0)


def der(fun, point, epsilon=np.sqrt(sys.float_info.epsilon)):
    return (fun(point + epsilon) - fun(point)) / epsilon


def check_gradient(fun, grad, R, dim, args=(), diff_eps=np.sqrt(sys.float_info.epsilon)):
    w = np.random.random(dim)
    w = (2 * w - 1) * R
    dw = np.eye(dim)
    g = grad(w, *args)
    norm = np.dot(g, g)
    difs = [np.abs((np.dot(g, dw_i) - der(lambda t: fun(w + t * dw_i, *args), 0, diff_eps))) for dw_i in dw]
    return np.average(difs) / norm


def check_hessian(grad, hess, R, dim, args=(), diff_eps=np.sqrt(sys.float_info.epsilon)):
    w = np.random.random(dim)
    w = (2 * w - 1) * R
    dw = np.eye(dim)
    h = hess(w, *args)
    norm = np.dot(h.flatten(), h.flatten())

    def xAy(A, x, y):
        return np.dot(x, np.dot(A, y))

    difs = [np.abs((xAy(h, dw1, dw2) - der(lambda t: np.dot(grad(w + t * dw1, *args), dw2), 0, diff_eps))) for dw1 in dw
            for dw2 in dw]
    return np.average(difs) / norm


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


def random_dataset(alpha, beta):
    xs = np.random.normal(size=(1000, alpha.shape[0]))
    labels = np.array([np.sign(np.dot(alpha, x) + beta) for x in xs])
    return xs, labels


a1a = sklearn.datasets.load_svmlight_file('data/a1a.txt')
X = a1a[0]
dummy = scipy.sparse.coo_matrix([[1] for i in range(X.shape[0])])
X_a1a = scipy.sparse.hstack([X, dummy])
labels_a1a = a1a[1]

breast_cancer = sklearn.datasets.load_svmlight_file('data/breast-cancer_scale.txt')
X = breast_cancer[0]
dummy = scipy.sparse.csr_matrix([[1] for i in range(X.shape[0])])
X_cancer = scipy.sparse.hstack([X, dummy])
labels_cancer = breast_cancer[1]-3

alpha = 2 * np.random.random(10) - 1
beta = 2 * np.random.random() - 1
X, labels_rand = random_dataset(alpha, beta)
dummy = scipy.sparse.csr_matrix([[1] for i in range(X.shape[0])])
X_rand = scipy.sparse.hstack([X, dummy])

X = X_cancer
labels = labels_cancer
X, X_test, labels, labels_test = train_test_split(X, labels, test_size=0.2)
labels_test = (labels_test + 1) / 2

print(check_gradient(function, gradient, 2, X.shape[1], args=[X, labels]))
print(check_hessian(gradient, hessian, 2, X.shape[1], args=[X, labels]))
exit(0)

w0 = 2 * np.random.random(X.shape[1]) - 1

# w_true = opt.minimize(function, w0, args=(X, labels), jac=gradient)['x']
# w_true = np.array([302.5925661, -21.69111231, 207.55842006, -411.28849642, -32.24866798
#                       , -4.31827468, -34.71080755, -73.71085179, 38.02308458, 49.91720216
#                       , 66.57753268, 15.52855102, 27.0831392, -312.8996407, 24.89967549
#                       , -34.07416117, 155.51644952, -106.31990291, 29.20772399, 65.4730172
#                       , 82.76286878, -19.74667188, 15.59046806, -451.20587395, 13.58897497
#                       , 79.36381994, -51.81633846, 19.5727145, -65.71161459, -102.40926345])
# print(np.mean(np.array([round(sigma(w_true, x)) for x in X_test]) == labels_test))
start = time.time()
w_res, iterations, oracles, times, accuracies, grad_ratios = optimization_task(function,
                                                                               gradient, w0,
                                                                               method='gradient descent',
                                                                               hess=hessian,
                                                                               args=[X, labels],
                                                                               one_dim_search='armiho',
                                                                               # search_kwargs=dict(
                                                                               # [('c', 0.5), ('x0', 10)]),
                                                                               # [('maxiter', 10)]),
                                                                               epsilon=0.00001,
                                                                               # true_min=function(w_true, X, labels)
                                                                               )

Xw = X_test.dot(w_res)
print(np.mean(np.array([round(sigmoid(p)) for p in Xw]) == labels_test))
print(np.linalg.norm(w_res))

# print('grad time', grad_time)
info(w_res, iterations, oracles, times, accuracies, grad_ratios, 0)

exit(0)
# graph(iterations[5:], accuracy[5:], title='1.1')
# graph(times[5:], accuracy[5:], title='2.1')
# graph(oracle[5:], accuracy[5:], title='3.1')
# graph(iterations[5:], grad_ratio[5:], title='1.2')
# graph(times[5:], grad_ratio[5:], title='2.2')
# graph(oracle[5:], grad_ratio[5:], title='3.2')

# print(*golden_search(polinome, -2, 2))
# minimization = opt.minimize_scalar(polinome)
# print(minimization['x'], minimization['fun'])
