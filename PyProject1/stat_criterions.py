import numpy as np
import matplotlib.pyplot as plt
import itertools

v17 = list(itertools.accumulate([7.92, 36.16, 25.09, 27.49, 3.34]))
v18 = list(itertools.accumulate([3.85, 31.13, 32.96, 29.92, 2.13]))

low17 = [0, 0, v17[0], v17[0], v17[1], v17[1], v17[2], v17[2], v17[3], v17[3], v17[4]]
high17 = [v17[0], v17[0], v17[1], v17[1], v17[2], v17[2], v17[3], v17[3], v17[4], v17[4], v17[4]]

low18 = [0, 0, v18[0], v18[0], v18[1], v18[1], v18[2], v18[2], v18[3], v18[3], v18[4]]
high18 = [v18[0], v18[0], v18[1], v18[1], v18[2], v18[2], v18[3], v18[3], v18[4], v18[4], v18[4]]

xs = [0, 20, 20, 40, 40, 60, 60, 80, 80, 100, 100]

plt.plot(xs, low17, 'r')
plt.plot(xs, high17, 'r')
plt.plot(xs, low18, 'g')
plt.plot(xs, high18, 'g')
plt.show()

sup = max([max(high17[i] - low18[i], high18[i] - low17[i]) for i in range(len(xs))]) / 100
print(np.sqrt(3.9 * 100000 / 2) * sup)
