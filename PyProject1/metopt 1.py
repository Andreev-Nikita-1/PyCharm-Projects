import numpy as np
import matplotlib.pyplot as plt
import sys


def xAx(x, A, y):
    return np.dot(x, np.dot(A, y))


def fun4_1(x, dx, A):
    return 0.5 * np.dot(x, x) ** 2 - xAx(x, A, x) + 0.5 * np.linalg.norm(A) ** 2


def der4_1(x, dx, A):
    return 2 * (np.dot(x, x) * np.dot(x, dx) - xAx(x, A, dx))


def gess4_1(x, dx, A):
    return 4 * np.dot(x, dx) ** 2 + 2 * np.dot(x, x) * np.dot(dx, dx) - 2 * np.dot(dx, np.dot(A, dx))


def fun4_2(x, dx, A):
    return xAx(x, A, x) / (np.dot(x, x))


def der4_2(x, dx, A):
    return 2 * (xAx(x, A, dx) / np.dot(x, x) - xAx(x, A, x) * np.dot(x, dx) / (np.dot(x, x) ** 2))


def gess4_2(x, dx, A):
    return (2 / (np.dot(x, x) ** 2)) * (
            (-4) * xAx(x, A, dx) * np.dot(x, dx) + 4 * xAx(x, A, x) * (np.dot(x, dx) ** 2) / np.dot(x, x) +
            xAx(dx, A, dx) * np.dot(x, x) - xAx(x, A, x) * np.dot(dx, dx))


def fun4_3(x, dx):
    return (np.dot(x, x) ** np.dot(x, x))


def der4_3(x, dx):
    return 2 * fun4_3(x, dx) * (np.log(np.dot(x, x)) + 1) * np.dot(x, dx)


def gess4_3(x, dx):
    xx = np.dot(x, x)
    return 4 * fun4_3(x, dx) * (
            ((np.log(xx) + 1) ** 2 + 1 / xx) * ((np.dot(x, dx)) ** 2) +
            (np.log(xx) + 1) * np.dot(dx, dx) / 2)


def fun5_1(x, dx):
    return np.trace(np.linalg.inv(x))


def der5_1(x, dx):
    invx = np.linalg.inv(x)
    return -np.trace(np.dot(np.dot(invx, dx), invx))


def gess5_1(x, dx):
    invx = np.linalg.inv(x)
    return 2 * np.trace(np.dot(np.dot(np.dot(invx, dx), invx), np.dot(dx, invx)))


def fun5_2(x, dx, v):
    return xAx(v, np.linalg.inv(x), v)


def der5_2(x, dx, v):
    invx = np.linalg.inv(x)
    return -xAx(v, np.dot(np.dot(invx, dx), invx), v)


def gess5_2(x, dx, v):
    invx = np.linalg.inv(x)
    return 2 * xAx(v, np.dot(np.dot(np.dot(invx, dx), invx), np.dot(dx, invx)), v)


def fun5_3(x, dx):
    n = len(x)
    return pow(np.abs((np.linalg.det(x))), 1 / n)


def der5_3(x, dx):
    n = len(x)
    return (1 / n) * fun5_3(x, dx) * np.trace(np.dot(np.linalg.inv(x), dx))


def gess5_3(x, dx):
    n = len(x)
    invx = np.linalg.inv(x)
    invxdx = np.dot(invx, dx)
    return fun5_3(x, dx) * (np.trace(invxdx / n) ** 2 - np.trace(np.dot(invxdx, invxdx) / n))


def fun6_1(v, dv):
    x = v[0]
    y = v[1]
    return 2 * x * x + y ** 2 * (x ** 2 - 2)


def der6_1(v, dv):
    x = v[0]
    y = v[1]
    df = [4 * x + 2 * y ** 2 * x, 2 * y * (x ** 2 - 2)]
    return np.dot(df, dv)


def gess6_1(v, dv):
    x = v[0]
    y = v[1]
    d2f = [[4 + 2 * y ** 2, 4 * x * y], [4 * x * y, 2 * (x ** 2 - 2)]]
    return xAx(dv, d2f, dv)


def fun6_2(v, dv, lam):
    x = v[0]
    y = v[1]
    return (1 - x) ** 2 + lam * (y - x ** 2) ** 2


def der6_2(v, dv, lam):
    x = v[0]
    y = v[1]
    df = [2 * (x - 1) + 4 * lam * (x ** 2 - y) * x, 2 * lam * (y - x ** 2)]
    return np.dot(df, dv)


def gess6_2(v, dv, lam):
    x = v[0]
    y = v[1]
    d2f = [[2 + 4 * lam * (3 * x ** 2 - y), -4 * lam * x], [-4 * lam * x, 2 * lam]]
    return xAx(dv, d2f, dv)


# численное дифференцирование
def der(function, point, epsilon):
    return (function(point + epsilon) - function(point)) / epsilon


def der32(function, point, epsilon):
    return np.float32(
        np.float32(function(np.float32(point) + np.float32(epsilon))) - np.float32(function(point))) / np.float32(
        epsilon)


# считает среднюю относительную точность численного дифференцирования по всем координатным осям в случайно выбранной точке
def randomcheck(fun, derv, R, dim, diff_eps=np.sqrt(sys.float_info.epsilon)):
    x = np.random.random((dim))
    x = (2 * x - 1) * R
    dx = np.eye(dim)
    dxs = np.array([[dx1 + dx2 for dx2 in dx] for dx1 in dx]).reshape(dim**2, dim)
    difs = [np.abs((derv(x, dx_i) - der(lambda t: fun(x + t * dx_i, dx_i), 0, diff_eps)) / derv(x, dx_i)) for dx_i in
            dxs]
    return np.average(difs)


# тоже самое для случая матриц
def randomcheck2(fun, derv, R, dim, diff_eps=np.sqrt(sys.float_info.epsilon)):
    x = np.random.random((dim, dim))
    x = (2 * x - 1) * R
    dx = np.eye(dim ** 2).reshape((dim ** 2, dim, dim))
    dxs = np.array([[dx1 + dx2 for dx2 in dx] for dx1 in dx]).reshape(dim ** 4, dim, dim)
    difs = [np.abs((derv(x, dx_i) - der(lambda t: fun(x + t * dx_i, dx_i), 0, diff_eps)) / derv(x, dx_i)) for dx_i in
            dxs]
    return np.average(difs)


def check_task4():
    n = 4
    A = 10 * np.random.random((n, n)) * np.eye(n)

    print()
    print("4.1")
    print(randomcheck(lambda x, dx: fun4_1(x, dx, A), lambda x, dx: der4_1(x, dx, A), 50, n))
    print(randomcheck(lambda x, dx: der4_1(x, dx, A), lambda x, dx: gess4_1(x, dx, A), 50, n))
    print()
    print("4.2")
    print(randomcheck(lambda x, dx: fun4_2(x, dx, A), lambda x, dx: der4_2(x, dx, A), 1, n))
    print(randomcheck(lambda x, dx: der4_2(x, dx, A), lambda x, dx: gess4_2(x, dx, A), 1, n))
    print()
    print("4.3")
    print(randomcheck(fun4_3, der4_3, 2, n))
    print(randomcheck(der4_3, gess4_3, 2, n))


def check_task5():
    n = 5

    print()
    print("5.1")
    print(randomcheck2(fun5_1, der5_1, 2, n))
    print(randomcheck2(der5_1, gess5_1, 2, n))

    print()
    print("5.2")
    v = 5 * np.random.random((n))
    print(randomcheck2(lambda x, dx: fun5_2(x, dx, v), lambda x, dx: der5_2(x, dx, v), 2, n))
    print(randomcheck2(lambda x, dx: der5_2(x, dx, v), lambda x, dx: gess5_2(x, dx, v), 2, n))

    print()
    print("5.3")
    print(randomcheck2(fun5_3, der5_3, 2, n))
    print(randomcheck2(der5_3, gess5_3, 2, n))


def check_task6():
    print()
    print("6 1")
    print(randomcheck(fun6_1, der6_1, 20, 2))
    print(randomcheck(der6_1, gess6_1, 20, 2))

    print()
    print("6 2")
    lam = 5 * (np.random.random() - 1)
    print(randomcheck(lambda x, dx: fun6_2(x, dx, lam), lambda x, dx: der6_2(x, dx, lam), 20, 2))
    print(randomcheck(lambda x, dx: der6_2(x, dx, lam), lambda x, dx: gess6_2(x, dx, lam), 20, 2))


def task7():
    def test_fun1(x):
        return x ** 3 - 4 * x ** 2 + 3 * x

    def test_der1(x):
        return 3 * x ** 2 - 8 * x + 3

    def test_fun2(x):
        return np.exp(x) + x ** 2

    def test_der2(x):
        return np.exp(x) + 2 * x

    def test_fun3(x):
        return np.log(1.1 + x)

    def test_der3(x):
        return 1 / (1.1 + x)

    def avg_diff(xs, fun, derv, eps, bit=64):
        if bit == 32:
            return np.average([np.abs(np.float32(derv(x)) - der32(fun, x, eps)) for x in xs])
        return np.average([np.abs(derv(x) - der(fun, x, eps)) for x in xs])

    def plot(epsilon, a, b, bit=64, filename=None, title=None):
        es = np.arange(a * epsilon, b * epsilon, b * epsilon / 1000)
        xs = np.arange(-1, 1, 0.1)
        ys1 = [avg_diff(xs, test_fun1, test_der1, eps, bit) for eps in es]
        ys2 = [avg_diff(xs, test_fun2, test_der2, eps, bit) for eps in es]
        ys3 = [avg_diff(xs, test_fun3, test_der3, eps, bit) for eps in es]
        fig, ax = plt.subplots()
        ax.plot(es, ys1, es, ys2, es, ys3)
        plt.axvline(epsilon)
        ax.set(title=title)
        ax.grid()
        # fig.savefig(filename)
        plt.show()

    eps64 = sys.float_info.epsilon
    plot(eps64, 1, 10, filename="metopt-eps.png", title="$\epsilon_m$   -   $10\epsilon_m$,    64bit")
    plot(np.sqrt(eps64), 0.05, 3, filename="metopt-sqrteps.png",
         title="$0.05\sqrt{\epsilon_m}$   -   $3\sqrt{\epsilon_m}$,    64bit")
    eps32 = 1.19209e-07
    plot(eps32, 1, 10, filename="metopt-eps32.png", title="$\epsilon_m$   -   $10\epsilon_m$,    32bit", bit=32)
    plot(np.sqrt(eps32), 0.05, 3, filename="metopt-sqrteps32.png",
         title="$0.05\sqrt{\epsilon_m}$   -   $3\sqrt{\epsilon_m}$,    32bit", bit=32)


check_task4()
check_task5()
check_task6()
# task7()
