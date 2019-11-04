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


def sigmoid(x):
    global sigmoids
    ans = 1 / (1 + np.exp(-x))
    sigmoids.append(ans)
    return ans


def oracle(w, X, labels, outers=None, order=0, grad_only=False):
    Xw = X.dot(w)
    sigmoids = [sigmoid(l * xw) for xw, l in zip(Xw, labels)]

    f = 0
    if not grad_only:
        f = -1 / X.shape[0] * np.sum([np.log(s) for s in sigmoids])

    if order == 0:
        return f

    grad_coeffs = np.array([l * (1 - s) for s, l in zip(sigmoids, labels)])
    X1 = X.multiply(grad_coeffs.reshape(-1, 1))
    g = -1 / X.shape[0] * np.array(X1.sum(axis=0)).reshape(X.shape[1])

    if order == 1:
        return f, g, 0

    hess_coeffs = np.array([s * (1 - s) for s in sigmoids])
    if outers is None:
        h = 1 / X.shape[0] * np.sum([np.outer(x, x) * hess_coeffs[i] for i, x in enumerate(X.todense())], axis=0)
    else:
        outers1 = outers.multiply(hess_coeffs.reshape(-1, 1))
        h = 1 / X.shape[0] * np.array(outers1.sum(axis=0)).reshape((X.shape[1], X.shape[1]))

    if order == 2:
        return f, g, h


def function(w, X, labels):
    return oracle(w, X, labels)


def gradient(w, X, labels):
    return oracle(w, X, labels, order=1)[1]


def hessian(w, X, labels):
    return oracle(w, X, labels, order=2)[2]


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


def golden_search_bounded(fun, a0, b0, eps=sys.float_info.epsilon, args=()):
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


def golden_search(fun, eps=sys.float_info.epsilon, args=()):
    a, _, b, _, _, _, onumber = opt.bracket(fun, args=args)
    if b < a:
        a, b = b, a
    # print(b-a, eps)
    gsb = golden_search_bounded(fun, a, b, eps=eps, args=args)
    return gsb[0], gsb[1] + onumber


def armiho(fun, c=0.1, k=10, f0=None, df0=None):
    x = 1
    oracle = 0

    if f0 == None:
        f0 = fun(0)
        oracle += 1

    if df0 is None:
        df0 = der(fun, 0)
        oracle += 2

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
            fkx = fun(x)
            oracle += 1
        else:
            break
    print(f0, "a>", fx)
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
    if np.linalg.matrix_rank(G) < G.shape[0]:
        G = G + 0.0001 * np.eye(G.shape[0])
    L = np.linalg.cholesky(G)
    Ltx = scipy.linalg.solve_triangular(L, d, lower=True)
    return scipy.linalg.solve_triangular(np.transpose(L), Ltx, lower=False)


def optimization_task(oracle, start, method='gradient descent', one_dim_search=None, args=(),
                      search_kwargs=dict([]), epsilon=0, max_iter=float('inf'), max_time=float('inf')):
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

    if one_dim_search == 'nester':
        L, L0 = 2, 0
        if 'L0' in search_kwargs.keys():
            L0 = 2 * search_kwargs['L0']
            L = L0

    while True:
        fk, gk, hk = oracle(x, *args, order=ord)
        fc += 1
        gc += 1
        if ord == 2:
            hc += 1
        if method == 'gradient descent':
            d = -gk
        elif method == 'newton':
            d = -solve(hk, gk)
        # elif solver == 'cg':
        #     d = -scipy.sparse.linalg.cg(hk, gk)[0]
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
            alpha, oracle_counter = armiho(fun, **search_kwargs, f0=fk, df0=np.dot(gk, d))
            print(fk, "->", fun(alpha))
            fc += oracle_counter
        elif one_dim_search == 'wolf':
            f_for_wolf = lambda z: oracle(z, *args)
            g_for_wolf = lambda z: oracle(z, *args, order=1, grad_only=True)[1]
            solution = opt.line_search(f_for_wolf, g_for_wolf, x, d, **search_kwargs, gfk=gk, old_fval=fk)
            alpha = solution[0]
            fc += solution[1]
            gc += solution[2]
        elif one_dim_search == 'nester':
            L, oracle_counter = nester(fun, fk, gk, L0=max(L / 2, L0))
            alpha = 1 / L
            fc += oracle_counter

        x = x + d * alpha

    print('-', end='')
    return x, iterations, times, values, grad_ratios, fcalls, gcalls, hcalls


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


def random_dataset(alpha, beta):
    xs = np.random.normal(size=(1000, alpha.shape[0]))
    labels = np.array([np.sign(np.dot(alpha, x) + beta) for x in xs])
    return xs, labels


c2, c3, c5 = [-0.95717287, -0.75630972, 0.72343828], [0.97490814, 0.80688161, 0.05165432, -0.52669906], [0.74540317,
                                                                                                         -0.50950527,
                                                                                                         0.96825081,
                                                                                                         -0.0243777,
                                                                                                         -0.33768442,
                                                                                                         -0.03532899]


# [-0.95717287 -0.75630972  0.72343828]
# [ 0.97490814  0.80688161  0.05165432 -0.52669906]
# [-0.22180215 -0.63603259  0.22768992  0.86681864 -0.32799243 -0.26688286]
# [ 0.74540317 -0.50950527  0.96825081 -0.0243777  -0.33768442 -0.03532899]
def pol2(x):
    return c2[0] + c2[1] * x + x ** 2


def pol3(x):
    return c3[0] + c3[1] * x + c3[2] * x ** 2 + c3[3] * x ** 3


def pol5(x):
    return c5[0] + c5[1] * x + c5[2] * x ** 2 + c5[3] * x ** 3 + c5[4] * x ** 4 + c5[5] * x ** 5


def pol(x):
    return -x ** 2


a1a = sklearn.datasets.load_svmlight_file('data/a1a.txt')
X = a1a[0]

dummy = scipy.sparse.coo_matrix([[1] for i in range(X.shape[0])])
X_a1a = scipy.sparse.hstack([X, dummy])
labels_a1a = a1a[1]

breast_cancer = sklearn.datasets.load_svmlight_file('data/breast-cancer_scale.txt')
X = breast_cancer[0]
dummy = scipy.sparse.csr_matrix([[1] for i in range(X.shape[0])])
X_cancer = scipy.sparse.hstack([X, dummy])
labels_cancer = breast_cancer[1] - 3

alpha = 2 * np.random.random(10) - 1
beta = 2 * np.random.random() - 1
X, labels_rand = random_dataset(alpha, beta)
dummy = scipy.sparse.csr_matrix([[1] for i in range(X.shape[0])])
X_rand = scipy.sparse.hstack([X, dummy])

outers_a1a = scipy.sparse.csr_matrix([np.outer(x, x).flatten() for x in X_a1a.todense()])
outers_cancer = scipy.sparse.csr_matrix([np.outer(x, x).flatten() for x in X_cancer.todense()])
outers_rand = scipy.sparse.csr_matrix([np.outer(x, x).flatten() for x in X_rand.todense()])

# w0_a1a = (2 * np.random.random(X_a1a.shape[1]) - 1) / 2
# X = X_cancer
# labels = labels_cancer
# X, X_test, labels, labels_test = train_test_split(X, labels, test_size=0.1)
# w0_cancer = (2 * np.random.random(X_cancer.shape[1]) - 1) / 2
# ww, _, _, t1, _, _, _, _ = optimization_task(oracle, w0_cancer, method='newton',
#                                              args=[X_cancer, labels_cancer, outers_cancer], epsilon=1e-8)
# Xw = X_test.dot(ww)
# labels_test = (labels_test + 1) / 2
# print(np.mean(np.array([round(sigmoid(p)) for p in Xw]) == labels_test))
# print(t1[-1])
# _, _, t2, _, _, _, _, _ = optimization_task(oracle, w0_a1a, method='newton', args=[X_a1a, labels_a1a, outers_a1a],
#                                            epsilon=1e-8)
# print(t2[-1])
# exit(0)
# X, X_test, labels, labels_test = train_test_split(X, labels, test_size=0.1)
# labels_test = (labels_test + 1) / 2

X = scipy.sparse.csr_matrix(X_a1a)
labels = labels_a1a

ind = [X[:, i].sum() for i in range(X.shape[1])]
offset = 0
for i in range(X.shape[1]):
    if ind[i] < 1:
        X1 = X[:, :(i - offset)]
        X2 = X[:, (i - offset + 1):]
        # X1_test = X_test[:, :(i - offset)]
        # X2_test = X_test[:, (i - offset + 1):]
        X = scipy.sparse.csr_matrix(scipy.sparse.hstack([X1, X2]))
        # X_test = scipy.sparse.csr_matrix(scipy.sparse.hstack([X1_test, X2_test]))
        offset += 1

# print(check_gradient(function, gradient, 2, X.shape[1], args=[X, labels]))
# print(check_hessian(gradient, hessian, 2, X.shape[1], args=[X, labels]))
# exit(0)

w0 = (2 * np.random.random(X.shape[1]) - 1) / 2
w0_a1a = (2 * np.random.random(X_a1a.shape[1]) - 1) / 2

# начальная точка

# w_true_a1a = opt.minimize(function, w0_a1a, args=(X_a1a, labels_a1a), jac=gradient)['x']
#
# f_min_a1a = function(w_true_a1a, X_a1a, labels_a1a)
#
w0_cancer = (2 * np.random.random(X_cancer.shape[1]) - 1) / 5


# # начальная точка
#
# w_true_cancer = opt.minimize(function, w0_cancer, args=(X_cancer, labels_cancer), jac=gradient)['x']
#
# f_min_cancer = function(w_true_cancer, X_cancer, labels_cancer)
#
# w0_rand = (2 * np.random.random(X_rand.shape[1]) - 1) / 2
#
# # начальная точка
#
# w_true_rand = opt.minimize(function, w0_rand, args=(X_rand, labels_rand), jac=gradient)['x']
#
# f_min_rand = function(w_true_rand, X_rand, labels_rand)


def graph_several(xs, ys, labels, x_l=None, y_l=None, title=None):
    fig, ax = plt.subplots()

    end = min([max(x) for x in xs])
    inds = [np.argmin([np.abs(p - end) for p in x]) for x in xs]
    xs1 = [xs[i][:inds[i]] for i in range(len(xs))]
    ys1 = [ys[i][:inds[i]] for i in range(len(xs))]

    def logs(y):
        return [np.log(v) for v in y]

    for i in range(len(xs)):
        ax.plot(xs1[i], ys1[i], label=labels[i])

    ax.set(xlabel=x_l, ylabel=y_l, title=title)
    ax.grid()
    ax.legend(fontsize=20)
    plt.show()


def info(w_res, iterations, oracle, times, accuracies, grad_ratios, w_true):
    print('|x_k - x*| =', np.linalg.norm(w_true - w_res))
    print('iterations =', iterations[-1])
    print('oracle =', oracle[-1])
    print('time =', times[-1])
    print('accuracy =', accuracies[-1])
    print('grad_ratio =', grad_ratios[-1])


a5, i5, t5, v5, r5, fc5, gc5, hc5 = optimization_task(oracle, w0_cancer, method='gradient descent',
                                                      args=[X_cancer, labels_cancer],
                                                      one_dim_search='armiho', max_time=1,
                                                      search_kwargs=dict([('k', 5)]))

graph_several([i5], [v5], ['a'])

exit(0)
a1, i1, o1, t1, v1, r1 = optimization_task(oracle, w0, solver='cholesky', method='newton', args=[X, labels],
                                           epsilon=0.0000001)
a1, i1, o1, t11, v1, r1 = optimization_task(oracle, w0_a1a, solver='cholesky', method='newton',
                                            args=[X_a1a, labels_a1a],
                                            epsilon=0.0000001)
print(t1[-1])
print(t11[-1])
#
# a2, i2, o2, t2, v2, r2 = optimization_task(oracle, w0_rand, method='gradient descent', args=[X_rand, labels_rand],
#                                            one_dim_search='golden', max_time=3, search_kwargs=dict([('eps', 0.1)]))
# a3, i3, o3, t3, v3, r3 = optimization_task(oracle, w0_rand, method='gradient descent', args=[X_rand, labels_rand],
#                                            one_dim_search='armiho', max_time=3, search_kwargs=dict([('x0', 50)]))
# a4, i4, o4, t4, v4, r4 = optimization_task(oracle, w0_rand, method=[]'gradient descent', args=[X_rand, labels_rand],
#                                            one_dim_search='wolf', max_time=3, search_kwargs=dict([('c2', 0.1)]))
# a5, i5, o5, t5, v5, r5 = optimization_task(oracle, w0_rand, method='gradient descent', args=[X_rand, labels_rand],
#                                            one_dim_search='nester', max_time=3)
# graph_several([t1, t2, t3, t4, t5], [r1, r2, r3, r4, r5],
#               labels=['brent', 'golden', 'armiho', 'wolf', 'nester'],
#               x_l='$time$', y_l='$|\nabla f(w_k)|^2/|\nabla f(w_0)|^2$')
# graph_several([i1, i2, i3, i4, i5], [r1, r2, r3, r4, r5],
#               labels=['brent', 'golden', 'armiho', 'wolf', 'nester'],
#               x_l='$time$', y_l='$|\nabla f(w_k)|^2/|\nabla f(w_0)|^2$')

# w2, i2, o2, t2, v2, r2 = optimization_task(oracle, w0, method='gradient descent', args=[X, labels],
#                                            one_dim_search='golden', max_time=5, search_kwargs=dict([('eps', 0.1)]))
# print(2)
# w3, i3, o3, t3, v3, r3 = optimization_task(oracle, w0, method='gradient descent', args=[X, labels],
#                                            one_dim_search='armiho', max_time=5, search_kwargs=dict([('x0', 10)]))
# print(3)
# w4, i4, o4, t4, v4, r4 = optimization_task(oracle, w0, method='gradient descent', args=[X, labels],
#                                            one_dim_search='wolf', max_time=5)
# print(4)
# w5, i5, o5, t5, v5, r5 = optimization_task(oracle, w0, method='gradient descent', args=[X, labels],
#                                            one_dim_search='nester', max_time=5)
# print(5)
# Xw = X_test.dot(w_)
# print(np.mean(np.array([round(sigmoid(p)) for p in Xw]) == labels_test))
# graph_several(t1, r1, t2, r2, t3, r3, t4, r4, t5, r5)
# info(w_res, iterations, oracles, times, accuracies, grad_ratios, 0)
# info(w_res1, iterations1, oracles1, times1, accuracies1, grad_ratios1, 0)
# print('grad time', grad_time)

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
# X = scipy.sparse.csr_matrix(X_a1a)
# labels=labels_a1a
# ind = [X[:, i].sum() for i in range(X.shape[1])]
# offset = 0
# for i in range(X.shape[1]):
#     if ind[i] < 1:
#         X1 = X[:, :(i - offset)]
#         X2 = X[:, (i - offset + 1):]
#         # X1_test = X_test[:, :(i - offset)]
#         # X2_test = X_test[:, (i - offset + 1):]
#         X = scipy.sparse.csr_matrix(scipy.sparse.hstack([X1, X2]))
#         # X_test = scipy.sparse.csr_matrix(scipy.sparse.hstack([X1_test, X2_test]))
#         offset += 1
# w0 = (2 * np.random.random(X.shape[1]) - 1)/3
# outers = scipy.sparse.csr_matrix([np.outer(x, x).flatten() for x in X.todense()])
# def solve(G, d):
#     return scipy.sparse.linalg.cg(G, d)[0]
