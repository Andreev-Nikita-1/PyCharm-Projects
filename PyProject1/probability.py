import numpy as np
import matplotlib.pyplot as plt
from math import factorial


class CKO:
    def __init__(self, type, teta, k, n=100):
        self.type = type
        self.teta = teta
        self.k = k
        self.n = n

    def generate(self, n=None):
        if n is None:
            n = self.n
        if self.type == "random":
            return np.random.rand(n) * self.teta
        if self.type == "exponential":
            return np.random.exponential(self.teta, n)

    def predict_teta(self, array=None):
        if array is None:
            array = self.generate()
        average = sum([x ** self.k for x in array]) / len(array)
        if self.type == "random":
            return ((self.k + 1) * average) ** (1 / self.k)
        if self.type == "exponential":
            return (average / factorial(self.k)) ** (1 / self.k)

    def CKO_value(self, r=100):
        sum = 0
        for i in range(r):
            sum += (self.teta - self.predict_teta()) ** 2
        return sum / r


n = 10000
r = 1000
teta = 10
k_min = 1
k_max = 10

y = []
for k in range(k_min, k_max):
    cko = CKO("random", teta, k, n)
    y.append(cko.CKO_value(r))
x = list(range(k_min, k_max))
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set(xlabel='k', ylabel='оценка СКО для равномерного',
       title=str(r) + ' выборок по ' + str(n) + " экземпляров, параметр = "+str(teta))
ax.grid()
# fig.savefig("randomCKO.png")
plt.show()


y = []
for k in range(k_min, k_max):
    cko = CKO("exponential", teta, k, n)
    y.append(cko.CKO_value(r))
x = list(range(k_min, k_max))
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set(xlabel='k', ylabel='оценка СКО для экспоненциального',
       title=str(r) + ' выборок по ' + str(n) + " экземпляров, параметр = "+str(teta))
ax.grid()
# fig.savefig("exponentialCKO.png")
plt.show()