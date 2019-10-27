import numpy as np
import matplotlib.pyplot as plt

a = 1
e_a = np.exp(-a)
c = 2 * a + 2 * e_a


def density(x):
    if np.abs(x) < a:
        return 1 / c
    else:
        return np.exp(-np.abs(x)) / c


def integral(x):
    if x < -a:
        return np.exp(x) / c
    elif x < a:
        return (e_a + a + x) / c
    else:
        return (2 * e_a + 2 * a - np.exp(-x)) / c


def inverse_integral(x):
    if x <= 0. or x >= 1.:
        return 0
    if x < e_a / c:
        return np.log(c * x)
    elif x < (e_a + 2 * a) / c:
        return c * x - e_a - a
    else:
        return -np.log(2 * e_a + 2 * a - c * x)


def generate_1():
    return inverse_integral(np.random.random())


def generate_2():
    x = np.random.exponential() * (1 - 2 * np.random.randint(0, 2))
    y = np.random.random() * np.exp(-np.abs(x)) / c / e_a
    while y > density(x):
        x = np.random.exponential() * (1 - 2 * np.random.randint(0, 2))
        y = np.random.random() * np.exp(-np.abs(x)) / c / e_a
    return x


def check_generator(n, generate):
    eps = 0.001
    R = 10
    xs = np.arange(-R, R, eps)
    xs = [np.round(x, decimals=3) for x in xs]
    count = dict(zip(xs, np.zeros(len(xs))))
    m = n / 100
    for i in range(n):
        if i % 10000 == 0:
            print(i / m, "%")
        x = generate()
        while x <= -R or x >= R:
            x = generate()
        count[np.round(x, decimals=3)] += 1 / n

    for i in range(len(xs) - 1):
        count[xs[i + 1]] += count[xs[i]]
    ys1 = np.array([count[i] for i in xs])
    ys2 = np.array([integral(i) for i in xs])
    fig, ax = plt.subplots()
    ax.plot(xs, ys1, xs, ys2)
    ax.set(title=str(generate) + ", " + str(n) + " экземпляров")
    ax.grid()
    plt.show()


check_generator(100000, generate_1)
check_generator(100000, generate_2)
