import numpy as np
import time
from .oracle import *
from .optimization_method import *


class Oracle_trashold:
    def __init__(self, oracle, l, t, start):
        self.oracle = oracle
        self.l = l
        self.t = t
        self.start = start
        self.fc = 0
        self.gc = 0
        self.hc = 0

    def reset_stats(self):
        self.oracle.reset_stats()
        self.update_stats()

    def update_stats(self):
        self.fc, self.gc, self.hc = self.oracle.fc, self.oracle.gc, self.oracle.hc

    def get_start(self):
        x = self.start
        u = np.abs(x) * 2 + 0.00001
        return np.hstack([x, u])

    def max_alpha(self, wu, dwu):
        w, u = np.array_split(wu, 2)
        dw, du = np.array_split(dwu, 2)
        inds1 = dw > du
        inds2 = dw + du < 0
        restr1 = -(u[inds1] - w[inds1]) / (du[inds1] - dw[inds1])
        restr2 = -(u[inds2] + w[inds2]) / (du[inds2] + dw[inds2])
        return np.min(np.hstack([restr1, restr2, 1e+100]))

    def evaluate(self, wu, order=0, no_function=False):
        w, u = np.array_split(wu, 2)
        g_, h_ = np.zeros(w.shape[0]), np.zeros(w.shape[0])
        if order == 0:
            f_ = self.oracle.evaluate(w, order=order, no_function=no_function)
        else:
            f_, g_, h_ = self.oracle.evaluate(w,
                                              order=order,
                                              no_function=no_function)

        self.update_stats()
        u2_w2 = (u ** 2 - w ** 2)
        trashold_f = -np.log(u2_w2).sum() / w.shape[0]
        trashold_g = -2 * np.hstack([-w, u]) / np.hstack([u2_w2, u2_w2])

        def trashold_h_mult(wu_):
            w_, u_ = np.array_split(wu, 2)
            w1 = 2 * (u ** 2 + w ** 2) / (u2_w2 ** 2) * w_ - 4 * w * u / (u2_w2 **
                                                                          2) * u_
            u1 = 2 * (u ** 2 + w ** 2) / (u2_w2 ** 2) * u_ - 4 * w * u / (u2_w2 **
                                                                          2) * w_
            return np.hstack([w1, u1])

        f = (f_ + self.l * u.sum()) + trashold_f / self.t
        g = (np.hstack([g_, self.l * np.ones(u.shape[0])]) + trashold_g / self.t)

        def hessian_mult(wu_):
            return np.hstack([h_(wu_[:w.shape[0]]), np.zeros(w.shape[0])]) + trashold_h_mult(wu_) / self.t

        if order == 0:
            return f
        return f, g, hessian_mult


def thrashold_method(oracle,
                     l,
                     eta,
                     method='L-BFGS',
                     max_iter=float('inf'),
                     max_time=float('inf'),
                     epsilon=1e-4):
    t = 100
    start = oracle.get_start()
    x = start
    start_time, k = time.time(), 0
    ratios, iterations, times = [], [], []
    _, g0, _ = oracle.evaluate(x, order=1, no_function=True)
    g0_norm = np.dot(g0 + l * np.sign(start), g0 + l * np.sign(start))
    while True:
        _, gk, _ = oracle.evaluate(x, order=1, no_function=True)
        subg = gk - np.sign(gk) * np.minimum(np.abs(gk),
                                             l) * (x == 0) + l * np.sign(x)
        ratio = subg.dot(subg) / g0_norm
        ratios.append(ratio)
        iterations.append(k)
        times.append(time.time() - start_time)

        if max_time < time.time() - start_time or max_iter < k:
            break
        k += 1
        print(k)
        oracle_t = Oracle_trashold(oracle, l=l, t=t, start=x)
        res = optimization_task(oracle_t,
                                method=method,
                                one_dim_search='armijo',
                                epsilon=epsilon,
                                max_iter=10,
                                max_time=max_time - (time.time() - start_time))
        print(k, '-')
        x = res['x'][:x.shape[0]]
        t *= eta

    return {'x': x, 'i': iterations, 'r': ratios, 't': times}

