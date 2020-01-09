import sys
import time
import numpy as np
import scipy.linalg
import scipy.optimize as opt


def golden_search_bounded(fun, a0, b0, eps=100 * sys.float_info.epsilon):
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


def golden_search(fun, eps=100 * sys.float_info.epsilon):
    a, b, _, _, _, _, _ = opt.bracket(fun)
    if b < a:
        a, b = b, a
    return golden_search_bounded(fun, a, b, eps=eps)


def armijo(fun, c=0.1, k=10, f0=None, df0=None, border=None):
    x = 1
    if border is not None and x > border:
        x = border / k ** 2

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
            if border is not None and x * k > border:
                break
            x *= k
            fx = fkx
            fkx = fun(k * x)
        else:
            break
    return x, fx


def solveCholesky(h, d, eps=0.0001):
    G = np.array([h(x) for x in np.eye(d.shape[0])])
    if np.linalg.matrix_rank(G) < G.shape[0]:
        G = G + eps * np.eye(G.shape[0])
    L = np.linalg.cholesky(G)
    Ltx = scipy.linalg.solve_triangular(L, d, lower=True)
    return scipy.linalg.solve_triangular(np.transpose(L), Ltx, lower=False)


def solveCG(h, b, eta=0.5, policy='sqrtGradNorm'):
    b_norm = np.linalg.norm(b)
    if policy == 'sqrtGradNorm':
        tol = min(np.sqrt(b_norm), 0.5)
    elif policy == 'gradNorm':
        tol = min(b_norm, 0.5)
    elif policy == 'const':
        tol = eta

    eps = tol * b_norm
    x0 = np.random.randn(b.shape[0]) * b_norm
    r = b - h(x0)
    p = r
    x = x0

    while True:
        rr = r.dot(r)
        hp = h(p)
        php = np.dot(p, hp)
        if php == 0:
            break
        alpha = rr / php
        x += alpha * p
        r -= alpha * hp
        if np.linalg.norm(r) < eps or rr == 0:
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
        if sty != 0:
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


def lipschitz(fun, fk, xk, gk, L0, l):
    L = L0
    while True:
        y = xk - 1 / L * gk
        y = np.sign(y) * np.maximum(np.abs(y) - l, 0)
        fx = fun(y)
        d = y - xk
        if fx < fk + np.dot(gk, d) + L / 2 * np.dot(d, d) or d.dot(d) == 0:
            break
        L *= 2
    return y, fx, L


def optimization_task(oracle,
                      method='gradient',
                      solver_kwargs={},
                      H0=None,
                      m=20,
                      one_dim_search=None,
                      search_kwargs={},
                      l=0,
                      epsilon=0,
                      max_iter=float('inf'),
                      max_time=float('inf')):
    # iterations, times, grad_ratios = [], [], []
    # values = []
    oracle.reset_stats()
    start_time, k = time.time(), 0
    start = oracle.get_start()
    x = start

    if method == 'gradient':
        ord = 1
        if one_dim_search is None:
            one_dim_search = 'armijo'
    elif method == 'newton' or method == 'hfn':
        ord = 2
        if one_dim_search is None:
            one_dim_search = 'constant step'
    elif method == 'BFGS' or method == 'L-BFGS':
        ord = 1
        init = True
        if one_dim_search is None:
            one_dim_search = 'constant step'
    elif method == 'l1prox':
        ord = 1
        one_dim_search = 'lipschitz'

    if one_dim_search == 'lipschitz':
        L, L0 = 2, 0
        if 'L0' in search_kwargs.keys():
            L0 = 2 * search_kwargs['L0']
            L = L0

    f0, g0, _ = oracle.evaluate(x, order=1)
    fk = f0
    g0_norm = np.dot(g0 + l * np.sign(start), g0 + l * np.sign(start))

    while True:

        f_, gk, hk = oracle.evaluate(x, order=ord, no_function=True)
        # values.append(fk + l * np.abs(x).sum())
        ratio = np.dot(gk, gk) / g0_norm
        if method == 'l1prox':
            subg = gk - np.sign(gk) * np.minimum(np.abs(gk), l) * (x == 0) + l * np.sign(x)
            ratio = subg.dot(subg) / g0_norm

        # iterations.append(k)
        # times.append(time.time() - start_time)
        # grad_ratios.append(ratio)

        if ratio <= epsilon or k > max_iter or time.time() - start_time > max_time:
            break

        k += 1

        if method == 'gradient':
            d = -gk
        elif method == 'hfn':
            d = solveCG(hk, -gk, **solver_kwargs)
        elif method == 'newton':
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
            alpha, fk = armijo(fun, **search_kwargs, f0=fk, df0=np.dot(gk, d), border=oracle.max_alpha())
        elif one_dim_search == 'wolfe':
            f_for_wolf = lambda z: oracle.evaluate(z)
            g_for_wolf = lambda z: oracle.evaluate(
                z, order=1, no_function=True)[1]
            solution = opt.line_search(f_for_wolf, g_for_wolf, x, d, **search_kwargs, gfk=gk, old_fval=fk)
            alpha = solution[0]
            if alpha is None:
                alpha = 1
            fk = solution[3]
            if fk is None:
                fk = fun(1)
        elif one_dim_search == 'lipschitz':
            x, fk, L = lipschitz(lambda z: oracle.evaluate(z), fk, x, gk, L0=max(L / 2, L0), l=l)
            continue
        else:
            alpha = one_dim_search(fun)

        if method == 'BFGS' or method == 'L-BFGS':
            prev_gk = gk
            prev_x = x

        x = x + d * alpha

    if one_dim_search != 'armijo' and one_dim_search != 'wolfe' and one_dim_search != 'lipschitz':
        fk = oracle.evaluate(x)
    return dict([('x', x), ('nit', k), ('fun', fk + l * np.abs(x).sum()), ('jac', gk),
                 ('time', time.time() - start_time), ('ratio', ratio), ('start', start),
                 ('nfev', oracle.fc), ('njev', oracle.gc), ('nhev', oracle.hc),
                 # ('i', iterations), ('t', times), ('r', grad_ratios),
                 # ('v', values),
                 ('null_components', (x == 0).sum())])


def SGD(oracle,
        alpha=0.1,
        harmonic_step=False,
        Q_reccurents=[],
        batch_size=20,
        max_iter=float('inf'),
        max_time=float('inf')):
    iterations, times, rec_values = [], [], []
    oracle.reset_stats()
    oracle.batch_size = batch_size
    oracle.reload()
    start_time, k = time.time(), 0
    start = oracle.get_start()
    x = start

    f0, g0, _ = oracle.evaluate(x, order=1)
    gk = g0
    Q = [f0 for _ in Q_reccurents]
    while True:

        rec_values.append(np.copy(Q))
        iterations.append(k)
        times.append(time.time() - start_time)
        if k > max_iter or time.time() - start_time > max_time:
            break

        k += 1
        if harmonic_step:
            x = x - gk * alpha / k
        else:
            x = x - gk * alpha
        fk, gk, _ = oracle.evaluate(x, order=1, no_function=True, stochastic=True)
        for i in range(len(Q_reccurents)):
            Q[i] = Q_reccurents[i](fk, Q[i])

    fun, jak, _ = oracle.evaluate(x, order=1)
    return dict([('x', x), ('nit', k), ('fun', fun), ('jac', jak), ('ratio', jak.dot(jak) / g0.dot(g0)),
                 ('time', time.time() - start_time), ('start', start), ('nfev', oracle.fc),
                 ('njev', oracle.gc), ('nhev', oracle.hc), ('i', iterations),
                 ('t', times), ('rec', rec_values), ('null_components', (x == 0).sum())])
