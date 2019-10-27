from sklearn.datasets import make_blobs, make_moons
import numpy as np
import pandas
import random
import matplotlib.pyplot as plt
import matplotlib
import math


def gini(x):
    labels = np.unique(x)
    gini = 1
    for label in labels:
        gini -= (x.tolist().count(label) / len(x)) ** 2
    return gini


def entropy(x):
    labels = np.unique(x)
    entropy = 0
    for label in labels:
        p = x.tolist().count(label) / len(x)
        entropy += - p * math.log(p, 2)
    return entropy


def gain_eval(left_y, right_y, criterion):
    n = len(left_y) + len(right_y)
    return criterion(np.concatenate([left_y, right_y])) - len(left_y) / n * criterion(left_y) - len(
        right_y) / n * criterion(right_y)


class DecisionTreeLeaf:
    def __init__(self, y, depth):
        self.depth = depth
        labels = np.unique(y)
        self.probs = dict([(l, y.tolist().count(l) / len(y)) for l in labels])
        self.y = max(self.probs.keys(), key=lambda k: self.probs[k])
        self.type = "leaf"


class DecisionTreeNode:
    def __init__(self, split_dim, split_value, left, right, depth):
        self.split_dim = split_dim
        self.split_value = split_value
        self.left = left
        self.right = right
        self.depth = depth
        self.type = "node"


class DecisionTreeClassifier:
    def __init__(self, criterion="gini", max_depth=None, min_samples_leaf=1):
        self.root = None
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def optimal_trashold(self, X, y, dim):
        Y = np.array(sorted(list(zip(X, y)), key=lambda x: x[0][dim]))
        optimal_trashold = 0
        max_gain = 0
        if self.criterion == "gini":
            optimal_trashold = self.min_samples_leaf + np.argmax(
                [gain_eval(Y[:k, 1], Y[k:, 1], gini) for k in
                 range(self.min_samples_leaf, len(X) - self.min_samples_leaf + 1)])
            max_gain = gain_eval(Y[:optimal_trashold, 1], Y[optimal_trashold:, 1], gini)
        if self.criterion == "entropy":
            optimal_trashold = self.min_samples_leaf + np.argmax(
                [gain_eval(Y[:k, 1], Y[k:, 1], entropy) for k in
                 range(self.min_samples_leaf, len(X) - self.min_samples_leaf + 1)])
            max_gain = gain_eval(Y[:optimal_trashold, 1], Y[optimal_trashold:, 1], entropy)
        return max_gain, Y[optimal_trashold][0][dim], Y[:optimal_trashold, 0], Y[:optimal_trashold, 1], Y[
                                                                                                        optimal_trashold:,
                                                                                                        0], Y[
                                                                                                            optimal_trashold:,
                                                                                                            1]

    def new_node(self, X, y, depth):
        if depth == self.max_depth or len(X) < 2 * self.min_samples_leaf:
            return DecisionTreeLeaf(y, depth)
        left = None
        left_labels = None
        right = None
        right_labels = None
        optimal_dim = 0
        trashold_value = 0
        max_gain = -1
        for dim in range(len(X[0])):
            gain, value, X_left, y_left, X_right, y_right = self.optimal_trashold(X, y, dim)
            if gain > max_gain:
                max_gain = gain
                optimal_dim = dim
                trashold_value = value
                left = X_left
                left_labels = y_left
                right = X_right
                right_labels = y_right
        if max_gain > 0:
            return DecisionTreeNode(optimal_dim, trashold_value, self.new_node(left, left_labels, depth + 1),
                                    self.new_node(right, right_labels, depth + 1), depth)
        else:
            return DecisionTreeLeaf(y, depth)

    def fit(self, X, y):
        self.root = self.new_node(X, y, 0)

    def find_explain(self, x, node):
        if node.type == "leaf":
            return "that is the reason"
        if node.type == "node":
            answer = "player's "
            if (node.split_dim == 0):
                answer += "monster kills per death"
            if (node.split_dim == 1):
                answer += "death in pvp percentage"
            if (node.split_dim == 2):
                answer += "player kills per death"
            if (node.split_dim == 3):
                answer += "accuracy"
            if x[node.split_dim] < node.split_value:
                answer += " is less then " + str(node.split_value) + ", and " + self.find_explain(x, node.left)
            else:
                answer += " is greater or equal then " + str(node.split_value) + ", and " + self.find_explain(x, node.right)
            return answer

    def find_x(self, x, node):
        if node.type == "leaf":
            return node.probs
        if node.type == "node":
            if x[node.split_dim] < node.split_value:
                return self.find_x(x, node.left)
            else:
                return self.find_x(x, node.right)

    def predict_proba(self, X):
        return [self.find_x(x, self.root) for x in X]

    def predict(self, X):
        proba = self.predict_proba(X)
        return [max(p.keys(), key=lambda k: p[k]) for p in proba]


def tree_depth(tree_root):
    if isinstance(tree_root, DecisionTreeNode):
        return max(tree_depth(tree_root.left), tree_depth(tree_root.right)) + 1
    else:
        return 1


def draw_tree_rec(tree_root, x_left, x_right, y):
    x_center = (x_right - x_left) / 2 + x_left
    if isinstance(tree_root, DecisionTreeNode):
        x_center = (x_right - x_left) / 2 + x_left
        x = draw_tree_rec(tree_root.left, x_left, x_center, y - 1)
        plt.plot((x_center, x), (y - 0.1, y - 0.9), c=(0, 0, 0))
        x = draw_tree_rec(tree_root.right, x_center, x_right, y - 1)
        plt.plot((x_center, x), (y - 0.1, y - 0.9), c=(0, 0, 0))
        plt.text(x_center, y, "x[%i] < %f" % (tree_root.split_dim, tree_root.split_value),
                 horizontalalignment='center')
    else:
        plt.text(x_center, y, str(tree_root.y),
                 horizontalalignment='center')
    return x_center


def draw_tree(tree, save_path=None):
    td = tree_depth(tree.root)
    plt.figure(figsize=(0.33 * 2 ** td, 2 * td))
    plt.xlim(-1, 1)
    plt.ylim(0.95, td + 0.05)
    plt.axis('off')
    draw_tree_rec(tree.root, -1, 1, td)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def plot_roc_curve(y_test, p_pred):
    positive_samples = sum(1 for y in y_test if y == 0)
    tpr = []
    fpr = []
    for w in np.arange(-0.01, 1.02, 0.01):
        y_pred = [(0 if p.get(0, 0) > w else 1) for p in p_pred]
        tpr.append(sum(1 for yp, yt in zip(y_pred, y_test) if yp == 0 and yt == 0) / positive_samples)
        fpr.append(sum(1 for yp, yt in zip(y_pred, y_test) if yp == 0 and yt != 0) / (len(y_test) - positive_samples))
    plt.figure(figsize=(7, 7))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.xlim(-0.01, 1.01)
    plt.ylim(-0.01, 1.01)
    plt.tight_layout()
    plt.show()


def rectangle_bounds(bounds):
    return ((bounds[0][0], bounds[0][0], bounds[0][1], bounds[0][1]),
            (bounds[1][0], bounds[1][1], bounds[1][1], bounds[1][0]))


def plot_2d_tree(tree_root, bounds, colors):
    if isinstance(tree_root, DecisionTreeNode):
        if tree_root.split_dim:
            plot_2d_tree(tree_root.left, [bounds[0], [bounds[1][0], tree_root.split_value]], colors)
            plot_2d_tree(tree_root.right, [bounds[0], [tree_root.split_value, bounds[1][1]]], colors)
            plt.plot(bounds[0], (tree_root.split_value, tree_root.split_value), c=(0, 0, 0))
        else:
            plot_2d_tree(tree_root.left, [[bounds[0][0], tree_root.split_value], bounds[1]], colors)
            plot_2d_tree(tree_root.right, [[tree_root.split_value, bounds[0][1]], bounds[1]], colors)
            plt.plot((tree_root.split_value, tree_root.split_value), bounds[1], c=(0, 0, 0))
    else:
        x, y = rectangle_bounds(bounds)
        plt.fill(x, y, c=colors[tree_root.y] + [0.2])


def plot_2d(tree, X, y, save_path=None):
    plt.figure(figsize=(9, 9))
    colors = dict((c, list(np.random.random(3))) for c in np.unique(y))
    bounds = list(zip(np.min(X, axis=0), np.max(X, axis=0)))
    plt.xlim(*bounds[0])
    plt.ylim(*bounds[1])
    plot_2d_tree(tree.root, list(zip(np.min(X, axis=0), np.max(X, axis=0))), colors)
    for c in np.unique(y):
        plt.scatter(X[y == c, 0], X[y == c, 1], c=[colors[c]], label=c)
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


# noise = 0.35
# X, y = make_moons(1500, noise=noise)
# X_test, y_test = make_moons(200, noise=noise)
# tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=15)
# tree.fit(X, y)
# plot_2d(tree, X, y, save_path="decision_tree_2d.png")
# plot_roc_curve(y_test, tree.predict_proba(X_test))
# draw_tree(tree, save_path="decision_tree.png")

# X, y = make_blobs(1500, 2, centers=[[0, 0], [-2.5, 0], [3, 2], [1.5, -2.0]])
# tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=30)
# tree.fit(X, y)
# plot_2d(tree, X, y)
# draw_tree(tree)


def predict_explain(dtc, X):
    return list(zip(dtc.predict(X), [dtc.find_explain(x, dtc.root) for x in X]))


def rak(label):
    if label == 'M':
        return 1
    else:
        return 0


def read_cancer_dataset(path_to_csv):
    dataframe = pandas.read_csv(path_to_csv, header=1)
    data = dataframe.values.tolist()
    np.random.shuffle(data)
    valuesArray = np.array(data)[:, 1:]
    labelsVector = np.array(data)[:, 0]
    labelsVector = np.array(list(map(rak, labelsVector)))
    valuesArray = np.array(list(map(lambda v: list(map(float, v)), valuesArray)))
    return valuesArray, labelsVector


def read_spam_dataset(path_to_csv):
    dataframe = pandas.read_csv(path_to_csv, header=1)
    data = dataframe.values.tolist()
    np.random.shuffle(data)
    valuesArray = np.array(data)[:, :-1]
    labelsVector = np.array(data)[:, -1]
    return valuesArray, labelsVector


def train_test_split(X, y, ratio):
    n = int(ratio * len(X))
    return X[:n], y[:n], X[n:], y[n:]


def read_dataset(path):
    dataframe = pandas.read_csv(path, header=1)
    dataset = dataframe.values.tolist()
    random.shuffle(dataset)
    y = [row[0] for row in dataset]
    X = [row[1:] for row in dataset]
    return np.array(X), np.array(y)


X, y = read_dataset("train.csv")
X_train, y_train, X_test, y_test = train_test_split(X, y, 0.9)
tree = DecisionTreeClassifier("entropy", max_depth=6, min_samples_leaf=5)
tree.fit(X_train, y_train)
print(len(list(filter(lambda x: x[0] != x[1], list(zip(tree.predict(X_test), y_test))))) / len(y_test))
# plot_roc_curve(y_test, tree.predict_proba(X_test))
# draw_tree(tree)

# X, y = read_cancer_dataset("cancer.csv")
# X_train, y_train, X_test, y_test = train_test_split(X, y, 0.9)


# dicts = tree.predict_proba(X_test)
# print(dicts[0])
# print(np.array([[dicts[i], y_test[i]] for i in range(len(X_test))]))
# print(list(filter(lambda x: x[0] != x[1], list(zip(tree.predict(X_test), y_test)))))
