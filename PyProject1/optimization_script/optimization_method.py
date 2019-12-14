import sys
import time
import numpy as np
import scipy.linalg
import scipy.optimize as opt


def golden_search_bounded(fun, a0, b0, eps=10 * sys.float_info.epsilon):
    ratio = (1 + 5 ** 0.5) / 2
    a, b, c, d = a0, b0, (b0 - a0) / ratio + a0, b0 - (b0 - a0) / ratio
    fc, fd = fun(c), fun(d)
    while True:
        if b - a <= eps:
            return c
        if fc < fd:
            c, d, b = a + d - c, c, d
            fd = fc
            fc = fun(c)
        else:
            a, c, d = c, d, b - d + c
            fc = fd
            fd = fun(d)


def golden_search(fun, eps=10 * sys.float_info.epsilon):
    a, _, b, _, _, _, onumber = opt.bracket(fun)
    if b < a:
        a, b = b, a
    return golden_search_bounded(fun, a, b, eps=eps)


def armijo(fun, c=0.1, k=10, f0=None, df0=None):
    x = 1

    if f0 is None:
        f0 = fun(0)

    fx = fun(x)
    fkx = fun(k * x)

    while True:
        if fx > f0 + c * df0 * x:
            x /= k
            fkx = fx
            fx = fun(x)
        elif fkx < f0 + k * c * df0 * x:
            x *= k
            fx = fkx
            fkx = fun(k * x)
        else:
            break

    return x, fx


def lipschitz(fun, fk, xk, gk, L0=1, l=1):
    L = L0
    while True:

        def prox(c):
            if c > l / L:
                return c - l / L
            elif c < -l / L:
                return c + l / L
            else:
                return 0

        y = xk - 1 / L * gk
        y = np.array([prox(y_i) for y_i in y])
        fx = fun(y)
        if fx < fk + np.dot(gk, y - xk) + L / 2 * np.dot(y - xk, y - xk):
            break
        L *= 2

    return y, fx


def solveCholesky(h, d, eps=0.0001):
    G = np.array([h(x) for x in np.eye(d.shape[0])])
    if np.linalg.matrix_rank(G) < G.shape[0]:
        G = G + eps * np.eye(G.shape[0])
    L = np.linalg.cholesky(G)
    Ltx = scipy.linalg.solve_triangular(L, d, lower=True)
    return scipy.linalg.solve_triangular(np.transpose(L), Ltx, lower=False)


def solveCG(h, b, eta=0.5, policy='sqrtGradNorm'):
    if eta is None:
        eta = 0.5
    if policy is None:
        policy = 'sqrtGradNorm'

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
        if np.linalg.norm(r) < eps:
            break
        beta = r.dot(r) / rr
        p = r + beta * p

    return x


class BFGS:
    def __init__(self, H0):
        self.Hk = H0

    def update(self, s, y):
        sty = np.dot(s, y)
        Hky = self.Hk.dot(y)
        sHky = np.outer(s, Hky)
        self.Hk += (sty + np.dot(y, Hky)) / sty ** 2 * np.outer(
            s, s) - (sHky + sHky.T) / sty


class LBFGS:
    def __init__(self, m):
        self.ys = []
        self.ss = []
        self.stys = []
        self.m = m

    def update(self, s, y):
        self.ss.append(s)
        self.ys.append(y)
        self.stys.append(np.dot(s, y))
        if len(self.ss) > self.m:
            self.ss = self.ss[1:]
            self.ys = self.ys[1:]
            self.stys = self.stys[1:]

    def direction(self, g):
        stgs = []
        for y, s, sty in list(zip(self.ys, self.ss, self.stys))[::-1]:
            stgs.append(np.dot(s, g))
            g -= y * stgs[-1] / sty
        g *= self.stys[-1] / np.dot(self.ys[-1], self.ys[-1])
        for y, s, sty, stg in zip(self.ys, self.ss, self.stys, stgs[::-1]):
            g += (-np.dot(y, g) + stg) / sty * s
        return g


def optimization_task(oracle, start, method='gradient descent', linear_solver='cg', solver_kwargs=dict([]), H0=None,
                      m=20, one_dim_search=None, search_kwargs=dict([]),
                      epsilon=0, max_iter=float('inf'), max_time=float('inf')):
    start_time, k = time.time(), 0
    x = start

    if method == 'gradient descent':
        ord = 1
        if one_dim_search is None:
            one_dim_search = 'armiho'
    elif method == 'newton':
        ord = 2
        if one_dim_search is None:
            one_dim_search = 'constant step'
    elif method == 'BFGS' or method == 'L-BFGS':
        ord = 1
        init = True
        if one_dim_search is None:
            one_dim_search = 'constant step'
    elif method == 'lasso':
        ord = 1
        one_dim_search = 'lipschitz'

    f0, g0, _ = oracle.evaluate(x, order=1)
    fk = f0

    if one_dim_search == 'lipschitz':
        L, L0, l = 2, 0, 0
        if 'l' in search_kwargs:
            l = search_kwargs['l']
        if 'L0' in search_kwargs.keys():
            L0 = 2 * search_kwargs['L0']
            L = L0

    while True:

        _, gk, hk = oracle.evaluate(x, order=ord, no_function=True)

        ratio = np.dot(gk, gk) / np.dot(g0, g0)

        if ratio <= epsilon or k > max_iter or time.time() - start_time > max_time:
            if one_dim_search != 'armijo' and one_dim_search != 'wolfe' and one_dim_search != 'lipschitz':
                fk = oracle.evaluate(x)
            break

        k += 1

        if method == 'gradient descent':
            d = -gk
        elif method == 'newton':
            if linear_solver == 'cg':
                d = solveCG(hk, -gk, **solver_kwargs)
            elif linear_solver == 'cholesky':
                d = solveCholesky(hk, -gk, **solver_kwargs)
        elif method == 'BFGS':
            if init:
                init = False
                bfgs = BFGS(H0 if H0 is not None else np.eye(x.shape[0]))
            else:
                bfgs.update(x - prev_x, gk - prev_gk)
            d = -bfgs.Hk.dot(gk)

        elif method == 'L-BFGS':
            if init:
                init = False
                lbfgs = LBFGS(m)
                d = -gk
            else:
                lbfgs.update(x - prev_x, gk - prev_gk)
                d = -lbfgs.direction(np.copy(gk))

        fun = lambda alpha: oracle.evaluate(x + d * alpha)

        if one_dim_search == 'constant step':
            alpha = 1
            if 'alpha' in search_kwargs.keys():
                alpha = search_kwargs['alpha']
        elif one_dim_search == 'golden_search':
            alpha = golden_search(fun, **search_kwargs)
        elif one_dim_search == 'brent':
            solution = opt.minimize_scalar(fun)
            alpha = solution['x']
        elif one_dim_search == 'armijo':
            alpha, fk = armijo(fun, **search_kwargs, f0=fk, df0=np.dot(gk, d))
        elif one_dim_search == 'wolfe':
            f_for_wolf = lambda z: oracle.evaluate(z)
            g_for_wolf = lambda z: oracle.evaluate(z, order=1, no_function=True)[1]
            solution = opt.line_search(f_for_wolf, g_for_wolf, x, d, **search_kwargs, gfk=gk, old_fval=fk)
            alpha = solution[0]
            if alpha is None:
                alpha = 1
            fk = solution[3]
            if fk is None:
                fk = fun(1)
        elif one_dim_search == 'lipschitz':
            x, fk = lipschitz(lambda z: oracle.evaluate(z), fk, gk, L0=max(L / 2, L0), l=l)
            continue
        else:
            alpha = one_dim_search(fun)

        if method == 'BFGS' or method == 'L-BFGS':
            prev_gk = gk
            prev_x = x

        x = x + d * alpha

    return dict([('x', x), ('nit', k), ('fun', fk), ('jac', gk), ('time', time.time() - start_time), ('ratio', ratio),
                 ('nfev', oracle.fc), ('njev', oracle.gc), ('nhev', oracle.hc)])
