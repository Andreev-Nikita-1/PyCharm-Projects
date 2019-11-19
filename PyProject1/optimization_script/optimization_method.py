import time
import numpy as np
import scipy.linalg
import scipy.optimize as opt


def golden_search_bounded(fun, a0, b0, eps=0.001, args=()):
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


def golden_search(fun, eps=0.001, args=()):
    a, _, b, _, _, _, onumber = opt.bracket(fun, args=args)
    if b < a:
        a, b = b, a
    gsb = golden_search_bounded(fun, a, b, eps=eps, args=args)
    return gsb[0], gsb[1] + onumber


def armijo(fun, c=0.1, k=10, f0=None, df0=None):
    x = 1
    oracle = 0

    if f0 is None:
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


def lipschitz(fun, f0, d, L0=1):
    L = L0
    oracle = 0
    fx = fun(1 / L)
    oracle += 1

    while fx > f0 - 1 / (2 * L) * np.dot(d, d):
        L *= 2
        fx = fun(1 / L)
        oracle += 1

    return L, fx, oracle


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
            return x
        beta = r.dot(r) / rr
        p = r + beta * p


def optimization_task(oracle, start, method='gradient descent', linear_solver='cg', solver_kwargs=dict([]),
                      one_dim_search=None, args=(), search_kwargs=dict([]),
                      epsilon=0, max_iter=float('inf'), max_time=float('inf')):
    start_time, k, fc, gc, hc = time.time(), 0, 0, 0, 0
    x = start

    if method == 'gradient descent':
        ord = 1
        if one_dim_search is None:
            one_dim_search = 'armijo'
    elif method == 'newton':
        ord = 2
        if one_dim_search is None:
            one_dim_search = 'unit step'

    f0, g0, _ = oracle(x, *args, order=1)
    fc += 1
    gc += 1
    fk = f0

    if one_dim_search == 'lipschitz':
        L, L0 = 2, 0
        if 'L0' in search_kwargs.keys():
            L0 = 2 * search_kwargs['L0']
            L = L0

    while True:
        _, gk, hk = oracle(x, *args, order=ord, no_function=True)
        gc += 1
        if ord == 2:
            hc += 1

        ratio = np.dot(gk, gk) / np.dot(g0, g0)

        if ratio <= epsilon or k > max_iter or time.time() - start_time > max_time:
            if one_dim_search != 'armijo' and one_dim_search != 'wolfe' and one_dim_search != 'lipschitz':
                fk = oracle(x, *args)
                fc += 1
            break

        k += 1

        if method == 'gradient descent':
            d = -gk
        elif method == 'newton':
            if linear_solver == 'cg':
                d = solveCG(hk, -gk, **solver_kwargs)
            elif linear_solver == 'cholesky':
                d = solveCholesky(hk, -gk, **solver_kwargs)

        fun = lambda alpha: oracle(x + d * alpha, *args)

        if one_dim_search == 'unit step':
            alpha = 1
        elif one_dim_search == 'golden_search':
            alpha, oracle_counter = golden_search(fun, **search_kwargs)
            fc += oracle_counter
        elif one_dim_search == 'brent':
            solution = opt.minimize_scalar(fun)
            alpha = solution['x']
            fc += solution['nfev']
        elif one_dim_search == 'armijo':
            alpha, fk, oracle_counter = armijo(fun, **search_kwargs, f0=fk, df0=np.dot(gk, d))
            fc += oracle_counter
        elif one_dim_search == 'wolfe':
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
        elif one_dim_search == 'lipschitz':
            L, fk, oracle_counter = lipschitz(fun, fk, gk, L0=max(L / 2, L0))
            alpha = 1 / L
            fc += oracle_counter
        else:
            alpha = one_dim_search(fun, *args)

        x = x + d * alpha

    return dict([('x', x), ('nit', k), ('fun', fk), ('jac', gk), ('time', time.time() - start_time), ('ratio', ratio),
                 ('nfev', fc), ('njev', gc), ('nhev', hc)])
