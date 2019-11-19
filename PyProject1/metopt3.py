import sys
import time
import matplotlib.pyplot as plt
import pylab
import numpy as np
import scipy
import scipy.optimize as opt
import scipy.sparse
import sklearn.datasets


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


def function(w, X, labels):
    return oracle(w, X, labels)


def gradient(w, X, labels):
    return oracle(w, X, labels, order=1, no_function=True)[1]


def hessian(w, X, labels):
    return oracle(w, X, labels, order=2, no_function=True)[2]


# численное дифференцирование
def der(fun, point, epsilon=np.sqrt(sys.float_info.epsilon)):
    return (fun(point + epsilon) - fun(point)) / epsilon


a1a = sklearn.datasets.load_svmlight_file('data/a1a.txt')
X = a1a[0]
dummy = scipy.sparse.csr_matrix([[1] for i in range(X.shape[0])])
X_a1a = scipy.sparse.hstack([X, dummy])
labels_a1a = a1a[1]

breast_cancer = sklearn.datasets.load_svmlight_file('data/breast-cancer_scale.txt')
X = breast_cancer[0]
dummy = scipy.sparse.csr_matrix([[1] for i in range(X.shape[0])])
X_cancer = scipy.sparse.hstack([X, dummy])
labels_cancer = breast_cancer[1] - 3


def random_dataset(alpha, beta):
    xs = np.random.normal(size=(1000, alpha.shape[0]))
    labels = np.array([np.sign(np.dot(alpha, x) + beta) for x in xs])
    return xs, labels


alpha = 2 * np.random.random(11) - 1
X, labels_rand = random_dataset(alpha[:-1], alpha[-1])
dummy = scipy.sparse.csr_matrix([[1] for i in range(X.shape[0])])
X_rand = scipy.sparse.hstack([X, dummy])


# на заданном интервале: возвращает точку минимума и число вызовов функции
def golden_search_bounded(fun, a0, b0, eps=100 * sys.float_info.epsilon, args=()):
    ratio = (1 + 5 ** 0.5) / 2
    a, b, c, d = a0, b0, (b0 - a0) / ratio + a0, b0 - (b0 - a0) / ratio
    fc, fd = fun(c, *args), fun(d, *args)
    onumber = 2
    while True:
        if b - a <= eps:
            return c, fc
        if fc < fd:
            c, d, b = a + d - c, c, d
            fd = fc
            fc = fun(c, *args)
            onumber += 1
        else:
            a, c, d = c, d, b - d + c
            fc = fd
            fd = fun(d, *args)
            onumber += 1


# общий случай используя bracket
def golden_search(fun, eps=100 * sys.float_info.epsilon, args=()):
    a, _, b, _, _, _, onumber = opt.bracket(fun, args=args)
    if b < a:
        a, b = b, a
    gsb = golden_search_bounded(fun, a, b, eps=eps, args=args)
    return gsb[0], gsb[1] + onumber


def armiho(fun, c=0.1, k=10, f0=None, df0=None):
    x = 1
    oracle = 0

    if f0 == None:
        f0 = fun(0)
        oracle += 1

    fx = fun(x)
    fkx = fun(k * x)
    oracle += 2

    while True:
        if fx > f0 + c * df0 * x:
            x /= k
            fkx = fx
            fx = fun(x)
            oracle += 1
        elif fkx < f0 + k * c * df0 * x:
            x *= k
            fx = fkx
            fkx = fun(k * x)
            oracle += 1
        else:
            break

    return x, fx, oracle


def nester(fun, f0, d, L0=1):
    L = L0
    oracle = 0
    fx = fun(1 / L)
    oracle += 1

    while fx > f0 - 1 / (2 * L) * np.dot(d, d):
        L *= 2
        fx = fun(1 / L)
        oracle += 1

    return L, fx, oracle


def solveCholesky(h, d, eps=0.0001, normHess=0):
    G = np.array([h(x) for x in np.eye(d.shape[0])])
    if normHess == 0:
        eps *= np.sqrt(np.linalg.norm(G.flatten())/ G.shape[0])
    elif normHess == 1:
        eps *= np.sqrt(np.linalg.norm(G.flatten()))
    if np.linalg.matrix_rank(G) < G.shape[0]:
        G = G + eps * np.eye(G.shape[0])
    L = np.linalg.cholesky(G)
    Ltx = scipy.linalg.solve_triangular(L, d, lower=True)
    return scipy.linalg.solve_triangular(np.transpose(L), Ltx, lower=False)


def solveCG(h, b, eta=0.01, policy='sqrtGradNorm'):
    b_norm = np.linalg.norm(b)
    if policy == 'sqrtGradNorm':
        tol = min(np.sqrt(b_norm), 0.5)
    elif policy == 'gradNorm':
        tol = min(b_norm, 0.5)
    elif policy == 'const':
        tol = eta

    eps = tol * b_norm
    x0 = np.random.random(b.shape[0]) * b_norm
    r = b - h(x0)
    p = r
    x = x0

    while True:
        rr = r.dot(r)
        hp = h(p)
        alpha = rr / np.dot(p, hp)
        x += alpha * p
        r -= alpha * hp
        # print(np.linalg.norm(r))
        if np.linalg.norm(r) < eps:
            return x
        beta = r.dot(r) / rr
        p = r + beta * p


def optimization_task(oracle, start, method='gradient descent', linear_solver='cg', solver_kwargs=dict([]),
                      one_dim_search=None, args=(), search_kwargs=dict([]),
                      epsilon=0, max_iter=float('inf'), max_time=float('inf')):
    iterations, fcalls, gcalls, hcalls, times, values, grad_ratios = [], [], [], [], [], [], []
    start_time, k, fc, gc, hc = time.time(), 0, 0, 0, 0
    x = start

    if method == 'gradient descent':
        ord = 1
        if one_dim_search is None:
            one_dim_search = 'brent'
    elif method == 'newton':
        ord = 2
        if one_dim_search is None:
            one_dim_search = 'unit step'

    f0, g0, _ = oracle(x, *args, order=1)
    fc += 1
    gc += 1
    fk = f0

    if one_dim_search == 'nester':
        L, L0 = 2, 0
        if 'L0' in search_kwargs.keys():
            L0 = 2 * search_kwargs['L0']
            L = L0

    flag = one_dim_search == 'armiho' or one_dim_search == 'wolf' or one_dim_search == 'nester'

    while True:
        f_, gk, hk = oracle(x, *args, order=ord, no_function=flag)
        # в реализации в __main__.py всегда стоит флаг no_function
        if not flag:
            fk = f_
            fc += 1
        gc += 1
        if ord == 2:
            hc += 1

        if method == 'gradient descent':
            d = -gk
        elif method == 'newton':
            if linear_solver == 'cg':
                d = solveCG(hk, -gk, **solver_kwargs)
            elif linear_solver == 'cholesky':
                d = solveCholesky(hk, -gk, **solver_kwargs)

        ratio = np.dot(gk, gk) / np.dot(g0, g0)
        iterations.append(k)
        k += 1
        fcalls.append(fc)
        gcalls.append(gc)
        hcalls.append(hc)
        times.append(time.time() - start_time)
        values.append(fk)
        grad_ratios.append(ratio)

        if ratio <= epsilon or k > max_iter or times[-1] > max_time:
            break

        fun = lambda alpha: oracle(x + d * alpha, *args)

        if one_dim_search == 'unit step':
            alpha = 1
        elif one_dim_search == 'golden':
            alpha, oracle_counter = golden_search(fun, **search_kwargs)
            fc += oracle_counter
        elif one_dim_search == 'brent':
            solution = opt.minimize_scalar(fun)
            alpha = solution['x']
            fc += solution['nfev']
        elif one_dim_search == 'armiho':
            alpha, fk, oracle_counter = armiho(fun, **search_kwargs, f0=fk, df0=np.dot(gk, d))
            fc += oracle_counter
        elif one_dim_search == 'wolf':
            f_for_wolf = lambda z: oracle(z, *args)
            g_for_wolf = lambda z: oracle(z, *args, order=1, no_function=True)[1]
            solution = opt.line_search(f_for_wolf, g_for_wolf, x, d, **search_kwargs, gfk=gk, old_fval=fk)
            alpha = solution[0]
            if alpha == None:
                alpha = 1
            fc += solution[1]
            gc += solution[2]
            fk = solution[3]
            if fk == None:
                fk = fun(1)
        elif one_dim_search == 'nester':
            L, fk, oracle_counter = nester(fun, fk, gk, L0=max(L / 2, L0))
            alpha = 1 / L
            fc += oracle_counter
        else:
            alpha = one_dim_search(fun, *args)

        x = x + d * alpha

    return x, iterations, times, values, grad_ratios, fcalls, gcalls, hcalls


def graph_several(xs, ys, labels, end=None, beg=0, x_l=None, y_l=None, title=None):
    pylab.rcParams['figure.figsize'] = 15, 7
    pylab.subplot(1, 1, 1)
    if end is None:
        end = max([max(x) for x in xs])
    ends = [np.argmin([np.abs(p - end) for p in x]) for x in xs]
    begs = [np.argmin([np.abs(p - beg) for p in x]) for x in xs]
    xs1 = [xs[i][begs[i]:ends[i] + 1] for i in range(len(xs))]
    ys1 = [ys[i][begs[i]:ends[i] + 1] for i in range(len(xs))]

    def logs(y):
        return [np.log(v) for v in y]

    for i in range(len(xs)):
        pylab.plot(xs1[i], logs(ys1[i]), label=labels[i])

    pylab.title(title)
    pylab.xlabel(x_l)
    pylab.ylabel(y_l)
    pylab.grid()
    pylab.legend()
    pylab.show()


w0_a1a = (2 * np.random.random(X_a1a.shape[1]) - 1) / 2
w0_cancer = (2 * np.random.random(X_cancer.shape[1]) - 1) / 2
w0_rand = (2 * np.random.random(X_rand.shape[1]) - 1) / 2

a1, i1, t1, v1, r1, fc1, gc1, hc1 = optimization_task(oracle, w0_cancer, method='newton',
                                                      args=[X_cancer, labels_cancer],
                                                      linear_solver='cholesky',
                                                      solver_kwargs=dict([('normHess', 0)]),
                                                      one_dim_search='unit step', max_time=0.1)
a2, i2, t2, v2, r2, fc2, gc2, hc2 = optimization_task(oracle, w0_cancer, method='newton',
                                                      args=[X_cancer, labels_cancer],
                                                      linear_solver='cholesky',
                                                      solver_kwargs=dict([('normHess', 1)]),
                                                      one_dim_search='unit step', max_time=0.1)
a3, i3, t3, v3, r3, fc3, gc3, hc3 = optimization_task(oracle, w0_cancer, method='newton',
                                                      args=[X_cancer, labels_cancer],
                                                      linear_solver='cholesky',
                                                      solver_kwargs=dict([('normHess', 2)]),
                                                      one_dim_search='unit step', max_time=0.1)

graph_several([t1, t2, t3], [r1, r2, r3], labels=['norm/n', 'norm', 'not norm'],
              x_l='time', y_l='gradNorm ratio')

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
