import time
import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import datasets
import copy


def visualize(X, labels_true, labels_pred, w):
    unique_labels = np.unique(labels_true)
    unique_colors = dict([(l, c) for l, c in zip(unique_labels, [[0.8, 0., 0.], [0., 0., 0.8]])])
    plt.figure(figsize=(9, 9))

    if w[1] == 0:
        plt.plot([X[:, 0].min(), X[:, 0].max()], w[0] / w[2])
    elif w[2] == 0:
        plt.plot(w[0] / w[1], [X[:, 1].min(), X[:, 1].max()])
    else:
        mins, maxs = X.min(axis=0), X.max(axis=0)
        pts = [[mins[0], -mins[0] * w[1] / w[2] - w[0] / w[2]],
               [maxs[0], -maxs[0] * w[1] / w[2] - w[0] / w[2]],
               [-mins[1] * w[2] / w[1] - w[0] / w[1], mins[1]],
               [-maxs[1] * w[2] / w[1] - w[0] / w[1], maxs[1]]]
        pts = [(x, y) for x, y in pts if mins[0] <= x <= maxs[0] and mins[1] <= y <= maxs[1]]
        x, y = list(zip(*pts))
        plt.plot(x, y, c=(0.75, 0.75, 0.75), linestyle="--")

    colors_inner = [unique_colors[l] for l in labels_true]
    colors_outer = [unique_colors[l] for l in labels_pred]
    plt.scatter(X[:, 0], X[:, 1], c=colors_inner, edgecolors=colors_outer)
    plt.savefig("data/perceptron.png")
    plt.show()


class Perceptron1:
    def __init__(self):
        self.w = None
        self.maxs = None
        self.mins = None
        self.offset = None

    def normalise(self, x):
        return (x - self.mins) / (self.maxs - self.mins)

    def fit(self, X, y):
        self.maxs = np.array([np.max(X[:, j]) for j in range(X.shape[1])])
        self.mins = np.array([np.min(X[:, j]) for j in range(X.shape[1])])
        Xs = np.array([self.normalise(x) for x in X])
        X1 = np.array([Xs[i] for i in range(len(Xs)) if y[i] == 1])
        X0 = np.array([Xs[i] for i in range(len(Xs)) if y[i] == 0])
        m0 = np.average(X0, axis=0)
        m1 = np.average(X1, axis=0)
        S0 = sum([np.outer(x - m0, x - m0) for x in X0])
        S1 = sum([np.outer(x - m1, x - m1) for x in X1])
        Sw = S0 + S1
        w = np.linalg.solve(Sw, m1 - m0)
        d = np.dot(m1 - m0, w)
        Xflat0 = np.array([np.dot(x - m0, w) / d for x in X0])
        Xflat1 = np.array([np.dot(x - m0, w) / d for x in X1])
        Xflat = list(zip(Xflat0, [0 for x in Xflat0]))
        Xflat.extend(list(zip(Xflat1, [1 for x in Xflat1])))
        Xflat.sort(key=lambda x: x[0])
        Xflat = np.array(Xflat)
        alpha = Xflat[0][0]
        wrong_num = len(Xflat0)
        min = wrong_num
        for i in range(len(Xflat)):
            if Xflat[i][1] == 0:
                wrong_num -= 1
            elif Xflat[i][1] == 1:
                wrong_num += 1
            if wrong_num < min:
                min = wrong_num
                alpha = (Xflat[i][0] + Xflat[i + 1][0]) / 2
        self.w = w
        self.offset = np.dot(alpha * m1 + (1 - alpha) * m0, w)

    def predict(self, X):
        return (1 + np.array([np.sign(np.dot(self.w, self.normalise(x)) - self.offset) for x in X])) / 2

    def getw(self):
        w = np.hstack(([-self.offset], self.w))
        w1 = w[1:] / (self.maxs - self.mins)
        return np.hstack(([w[0] - np.dot(w1, self.mins)], w1))


class Perceptron:
    def __init__(self, iterations=100):
        self.iterations = iterations
        self.X = None
        self.y = None
        self.w = None
        self.maxs = None
        self.mins = None
        self.besterror = None
        self.bestw = None

    def iteration(self):
        wrongs = [i for i in range(len(self.X)) if self.y[i] * np.dot(self.w, self.X[i]) < 0]
        if len(wrongs) < self.besterror:
            self.besterror = len(wrongs)
            self.bestw = self.w
        index = np.random.choice(wrongs)
        self.w = self.w + self.y[index] * self.X[index]

    def normalise(self, x):
        return np.hstack(([1], (x - self.mins) / (self.maxs - self.mins)))

    def fit(self, X, y):
        self.maxs = np.array([np.max(X[:, j]) for j in range(X.shape[1])])
        self.mins = np.array([np.min(X[:, j]) for j in range(X.shape[1])])
        self.X = np.array(
            [self.normalise(x) for x in X])
        self.y = (y * 2) - 1
        self.w = np.random.random((X.shape[1] + 1))
        self.bestw = self.w
        self.besterror = len(X)
        for i in range(self.iterations):
            self.iteration()
        self.w = self.bestw

    def predict(self, X):
        return (1 + np.array([np.sign(np.dot(self.w, self.normalise(x))) for x in X])) / 2

    def getw(self):
        w1 = self.w[1:] / (self.maxs - self.mins)
        return np.hstack(([self.w[0] - np.dot(w1, self.mins)], w1))


def first_feature(image):
    rounded = np.array([[round(p) for p in v] for v in image])
    count = np.array([rounded[:, j].tolist().count(1) for j in range(8)]) / 8
    weights = [0.0, 0.1, 0.2, 0.2, 0.2, 0.2, 0.1, 0.0]
    return np.dot(weights, 4 * count * (1 - count))


def second_feature(image):
    lengths = np.array([np.sum(v) for v in image])
    var = 0
    for i in range(7):
        var += np.abs(lengths[i] - lengths[i + 1])
    return var


def third_feature(image):
    top = image[1:4]
    bottom = image[4:8]

    def mass_center(ar):
        m_c = 0
        s = 0
        for i in range(8):
            m_c += i * (ar[0][i] + ar[1][i] + ar[2][i])
            s += ar[0][i] + ar[1][i] + ar[2][i]
        return m_c / s

    return mass_center(bottom) - mass_center(top)


def fourth_feature(image):
    sum = 0
    rounded = np.array([[round(p) for p in v] for v in image])
    for j in range(8):
        for i in range(8):
            s1 = np.sum(rounded[:i, j])
            s2 = np.sum(rounded[i + 1:, j])
            sum += (1 - rounded[i, j]) * s1 * s2
    return sum


def transform_images(images):
    return np.array([[first_feature(x), fourth_feature(x)] for x in images])


data = datasets.load_digits()
images, labels = data.images, data.target
mask = np.logical_or(labels == 1, labels == 5)
labels = labels[mask]
labels = np.array([(x - 1) // 4 for x in labels])
images = images[mask]
images /= np.max(images)
X = transform_images(images)

X_train, X_test, y_train, y_test = train_test_split(X, labels, train_size=0.8, shuffle=False)

c = Perceptron(iterations=100000)
# c = Perceptron1()
c.fit(X_train, y_train)
# wrongs = len(X_train) + np.array([i for i in range(len(X_test)) if y_test[i] != c.predict([X_test[i]])[0]])
# wrong_images = images[wrongs]
# wrong_X = X[wrongs]
# wrong_y = labels[wrongs]
# #
# visualize(wrong_X, wrong_y, np.array(c.predict(wrong_X)), c.getw())
visualize(X_train, y_train, np.array(c.predict(X_train)), c.getw())
# visualize(X_test, y_test, np.array(c.predict(X_test)), c.getw())
print("Accuracy:", np.mean(c.predict(X_test) == y_test))

# X, true_labels = make_blobs(400, 2, centers=[[0, 0], [2.5, 2.5]])
# c = Perceptron(iterations=100)
# c1 = Perceptron1()
# start = time.time()
# c1.fit(X, true_labels)
# print('time', time.time() - start)
# print(np.mean(true_labels == c1.predict(X)))
# visualize(X, true_labels, np.array(c1.predict(X)), c1.getw())
# start = time.time()
# c.fit(X, true_labels)
# print('time', time.time() - start)
# print(np.mean(true_labels == c.predict(X)))
# visualize(X, true_labels, np.array(c.predict(X)), c.getw())
# X, true_labels = make_moons(400, noise=0.075)
# c.fit(X, true_labels)
# c1.fit(X, true_labels)
# print(np.mean(true_labels == c.predict(X)))
# visualize(X, true_labels, np.array(c.predict(X)), c.getw())
# print(np.mean(true_labels == c1.predict(X)))
# visualize(X, true_labels, np.array(c1.predict(X)), c1.getw())
