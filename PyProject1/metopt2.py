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


def oracle(w, X, labels, outers=None, order=0):
    Xw = X.dot(w)
    sigmoids = [sigmoid(l * xw) for xw, l in zip(Xw, labels)]
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


def optimization_task(oracle, start, method='gradient descent', one_dim_search=None, args=(),
                      search_kwargs=dict([]), epsilon=0.0001):
    iterations, oracles, times, values, grad_ratios = [], [], [], [], []
    start_time, k, onumber = time.time(), 0, 0
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
    onumber += 1

    if one_dim_search == 'nester':
        L, L0 = 2, 0
        if 'L0' in search_kwargs.keys():
            L0 = 2 * search_kwargs['L0']
            L = L0

    while True:
        fk, gk, hk = oracle(x, *args, order=ord)
        if method == 'gradient descent':
            d = -gk
        elif method == 'newton':
            # d = -1*solve(1*hk, gk)
            det = np.linalg.det(hk)
            inv = np.linalg.inv(hk)
            id = np.dot(hk, inv)
            d = -np.linalg.solve(hk, gk)
        onumber += 1
        ratio = np.dot(gk, gk) / np.dot(g0, g0)
        iterations.append(k)
        k += 1
        oracles.append(onumber)
        times.append(time.time() - start_time)
        values.append(fk)
        grad_ratios.append(ratio)

        if ratio <= epsilon:
            break

        fun = lambda alpha: oracle(x + d * alpha, *args)

        if one_dim_search == 'unit step':
            alpha = 1
        elif one_dim_search == 'golden':
            alpha, oracle_counter = golden_search(fun, **search_kwargs)
            onumber += oracle_counter
        elif one_dim_search == 'brent':
            solution = opt.minimize_scalar(fun)
            alpha = solution['x']
            onumber += solution['nfev']
        elif one_dim_search == 'armiho':
            alpha, oracle_counter = armiho(fun, **search_kwargs, df0=np.dot(gk, d))
            onumber += oracle_counter
        elif one_dim_search == 'wolf':
            f_for_wolf = lambda z: oracle(z, *args)
            g_for_wolf = lambda z: oracle(z, *args, order=1)[1]
            solution = opt.line_search(f_for_wolf, g_for_wolf, x, d, **search_kwargs, gfk=gk, old_fval=fk)
            alpha = solution[0]
            # wolf не принимает оракул, вычисляющий одновременно функцию и градиент, хотя мог бы,
            # так что я учёл только вызовы функции
            onumber += solution[1]
        elif one_dim_search == 'nester':
            L, oracle_counter = nester(fun, fk, gk, L0=max(L / 2, L0))
            alpha = 1 / L
            onumber += oracle_counter

        x = x + d * alpha

    return x, iterations, oracles, times, values, grad_ratios


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

ind = [X[:, i].sum() for i in range(X.shape[1])]
offset = 0
for i in range(X.shape[1]):
    if ind[i] < 1:
        X1 = X[:, :(i - offset)]
        X2 = X[:, (i - offset + 1):]
        X = scipy.sparse.csr_matrix(scipy.sparse.hstack([X1, X2]))
        offset += 1

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

X = X_a1a
labels = labels_a1a
X, X_test, labels, labels_test = train_test_split(X, labels, test_size=0.2)
labels_test = (labels_test + 1) / 2

# print(check_gradient(function, gradient, 2, X.shape[1], args=[X, labels]))
# print(check_hessian(gradient, hessian, 2, X.shape[1], args=[X, labels]))
# exit(0)
w0 = (2 * np.random.random(X.shape[1]) - 1) / 2
w0 = np.array([-0.22782837460727068, -0.002099347152515363, -0.052140320146127794, -0.4594284595239546, -0.01790283237206236,
        0.12983476020225782, 0.06740164787764125, -0.3337806990901412, -0.49798962838309047, 0.4967946291937556,
        0.49947796881425977, -0.09544983781315464, 0.35246564883124387, 0.042102009685434316, 0.03763645270942928,
        0.4197768498715374, 0.30686599450997565, -0.4088194254895058, 0.01089880869436477, 0.11074770204517437,
        0.1458653004565148, 0.208446036593952, 0.39441828933005285, -0.04954012997882096, 0.3997810996111618,
        0.04175069070997195, -0.023943010348827176, 0.32085419857798136, 0.4122901348096297, -0.397076808619621,
        -0.1534729352348143, -0.4512259695905094, 0.41267803474570397, 0.3266596907739706, -0.03310330731526023,
        -0.06666283353349534, -0.4210292318993487, -0.25464195106956045, -0.3199620612488415, 0.15099360453960298,
        -0.3097065319705129, -0.10117963593951385, -0.2717171466026035, -0.4549372062720147, 0.28360061021875826,
        -0.3888476541559168, -0.35662416622919657, -0.4036774006681443, -0.4744773827292663, 0.2546036767889589,
        0.055721289208309766, -0.2283143241680874, -0.4584855699794592, 0.1713218241748673, 0.2620666565700679,
        0.26266780648561083, 0.16764972456129468, -0.10932321124723621, -0.385458180352781, -0.38758038413679197,
        0.05746032733818762, -0.17022170536498382, -0.05791703393282366, 0.25537223313255, 0.1585573750515611,
        -0.4101317974960528, 0.17155427728164663, -0.4036483353011906, 0.1815284371776572, 0.4385478415143389,
        0.3694434971273496, -0.4685203817255963, 0.1998003525238219, -0.303425514916411, 0.37438376301236054,
        0.3253888620624308, -0.06188391448218611, 0.20414547021540097, -0.4447208289352087, 0.42277682164314456,
        -0.21885073294917856, -0.4659017885192317, -0.2758820705064968, 0.01997386445525684, 0.14604893492472004,
        -0.04241116506593112, -0.1251896834662446, -0.16547884227699117, 0.2541978375077075, 0.32854906459749966,
        0.4677698834363333, 0.2400369797678561, 0.22852724155878756, -0.40874046465244807, 0.45920375858258067,
        -0.4929731298148555, -0.3484696509284765, 0.25647268563669756, 0.11059872940640958, 0.37811934277832115,
        -0.48144838637311216, -0.19321482313146854, -0.42316752861272255, -0.1375946350791699, 0.27015633616229107,
        -0.1920931320570013, 0.08308477880887266, -0.31502981042925304, 0.18725445145397945, -0.08422207550990368,
        0.03609274593742673, 0.4465238370255289, -0.25264457814011465, -0.23830625860277765])

# w_true = opt.minimize(function, w0, args=(X, labels), jac=gradient)['x']
# w_true = np.array([302.5925661, -21.69111231, 207.55842006, -411.28849642, -32.24866798
#                       , -4.31827468, -34.71080755, -73.71085179, 38.02308458, 49.91720216
#                       , 66.57753268, 15.52855102, 27.0831392, -312.8996407, 24.89967549
#                       , -34.07416117, 155.51644952, -106.31990291, 29.20772399, 65.4730172
#                       , 82.76286878, -19.74667188, 15.59046806, -451.20587395, 13.58897497
#                       , 79.36381994, -51.81633846, 19.5727145, -65.71161459, -102.40926345])
# print(np.mean(np.array([round(sigma(w_true, x)) for x in X_test]) == labels_test))

outers = np.array([np.outer(x, x).flatten() for x in X.todense()])
outers = scipy.sparse.csr_matrix(outers)
start = time.time()
w_res, iterations, oracles, times, accuracies, grad_ratios = optimization_task(oracle, w0,
                                                                               method='newton',
                                                                               args=[X, labels, outers],
                                                                               # one_dim_search='armiho',
                                                                               # search_kwargs=dict(
                                                                               # [('c', 0.5), ('x0', 10)]),
                                                                               # [('maxiter', 10)]),
                                                                               epsilon=0.000000001
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
