import sys
import time
import numpy as np
import scipy
import scipy.optimize as opt
import scipy.sparse
import sklearn.datasets


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def oracle(w, X, labels, outers=None, order=0, no_function=False):
    Xw = X.dot(w)
    sigmoids = [sigmoid(l * xw) for xw, l in zip(Xw, labels)]

    f = 0
    if not no_function:
        f = -1 / X.shape[0] * np.sum([np.log(s) for s in sigmoids])

    if order == 0:
        return f

    grad_coeffs = np.array([l * (1 - s) for s, l in zip(sigmoids, labels)])
    X1 = X.multiply(grad_coeffs.reshape(-1, 1))
    g = -1 / X.shape[0] * np.array(X1.sum(axis=0)).reshape(X.shape[1])

    if order == 1:
        return f, g, 0

    hess_coeffs = np.array([s * (1 - s) for s in sigmoids])
    # outers - массив x x^t flatten, который можно посчитать заранее
    if outers is None:
        h = 1 / X.shape[0] * np.sum([np.outer(x, x) * hess_coeffs[i] for i, x in enumerate(X.todense())], axis=0)
    else:
        outers1 = outers.multiply(hess_coeffs.reshape(-1, 1))
        h = 1 / X.shape[0] * np.array(outers1.sum(axis=0)).reshape((X.shape[1], X.shape[1]))

    if order == 2:
        return f, g, h


# на заданном интервале: возвращает точку минимума и число вызовов функции
def golden_search_bounded(fun, a0, b0, eps=100 * sys.float_info.epsilon, args=()):
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


def solve(G, d):
    if np.linalg.matrix_rank(G) < G.shape[0]:
        G = G + 0.0001 * np.eye(G.shape[0])
    L = np.linalg.cholesky(G)
    Ltx = scipy.linalg.solve_triangular(L, d, lower=True)
    return scipy.linalg.solve_triangular(np.transpose(L), Ltx, lower=False)


def optimization_task(oracle, start, method='gradient descent', one_dim_search=None, args=(),
                      search_kwargs=dict([]), epsilon=0, max_iter=float('inf'), max_time=float('inf')):
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

    while True:
        _, gk, hk = oracle(x, *args, order=ord, no_function=True)
        gc += 1
        if ord == 2:
            hc += 1

        if method == 'gradient descent':
            d = -gk
        elif method == 'newton':
            d = -solve(hk, gk)

        ratio = np.dot(gk, gk) / np.dot(g0, g0)
        k += 1

        if ratio <= epsilon or k > max_iter or time.time() - start_time > max_time:
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

    return x, oracle(x, *args), k, ratio, time.time() - start_time, fc, gc, hc


def main():
    method = 'gradient descent'
    one_dim_search = 'wolf'
    seed = 0
    epsilon = 0.00001
    distr = 'uniform'

    for arg in sys.argv[1:]:
        name, value = arg.split('=')
        if name == '--ds_path':
            path = value
        elif name == '--optimize_method':
            if value == 'gradient':
                method = 'gradient descent'
            elif value == 'newton':
                method = 'newton'
        elif name == '--line_search':
            if value == 'golden_search':
                one_dim_search = 'golden'
            elif value == 'brent':
                one_dim_search = 'brent'
            elif value == 'armojo':
                one_dim_search = 'armiho'
            elif value == 'wolfe':
                one_dim_search = 'wolf'
            elif value == 'lipschitz':
                one_dim_search = 'nester'
        elif name == '--seed':
            seed = value
        elif name == '--eps':
            epsilon = float(value)
        elif name == '--point_distribution':
            if value == 'uniform':
                distr = 'uniform'
            else:
                distr = 'normal'

    data = sklearn.datasets.load_svmlight_file(path)
    X = data[0]
    dummy = scipy.sparse.csr_matrix([[1] for i in range(X.shape[0])])
    X = scipy.sparse.hstack([X, dummy])
    labels = data[1]
    d = dict([(l, 2 * y - 1) for y, l in enumerate(np.unique(labels))])
    labels = np.array([d[l] for l in labels])

    np.random.seed(int(seed))
    if distr == 'uniform':
        w0 = (2 * np.random.random(X.shape[1]) - 1) / 2
    else:
        w0 = np.random.normal(0, 0.5, X.shape[1])

    outers = scipy.sparse.csr_matrix([np.outer(x, x).flatten() for x in X.todense()])

    w, fw, k, r, t, fc, gc, hc = optimization_task(oracle, w0, method=method, one_dim_search=one_dim_search,
                                                   args=[X, labels, outers], epsilon=epsilon)
    answer = '{\n' + \
             '\t \'initial_point\': ' + '\'' + str(w0) + '\',\n' + \
             '\t \'optimal_point\': ' + '\'' + str(w) + '\',\n' + \
             '\t \'function_value\': ' + '\'' + str(fw) + '\',\n' + \
             '\t \'function_calls\': ' + '\'' + str(fc) + '\',\n' + \
             '\t \'gradient_calls\': ' + '\'' + str(gc) + '\',\n' + \
             '\t \'hessian_calls\': ' + '\'' + str(hc) + '\',\n' + \
             '\t \'r_k\': ' + '\'' + str(r) + '\',\n' + \
             '\t \'working_time\': ' + '\'' + str(t) + '\'\n' + \
             '}'

    print(answer)


if __name__ == '__main__':
    main()
