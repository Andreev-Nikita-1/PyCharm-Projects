from sklearn.model_selection import train_test_split
import numpy as np
import pandas
import random
import matplotlib.pyplot as plt
import matplotlib
import copy
from catboost import CatBoostClassifier, Pool
from math import sqrt
import time


def read_dataset(path):
    dataframe = pandas.read_csv(path, header=0)
    dataset = dataframe.values.tolist()
    random.shuffle(dataset)
    y_age = [row[0] for row in dataset]
    y_sex = [row[1] for row in dataset]
    X = [row[2:] for row in dataset]
    return np.array(X), np.array(y_age), np.array(y_sex), list(dataframe.columns)[2:]


def gini(x):
    _, counts = np.unique(x, return_counts=True)
    proba = counts / len(x)
    return np.sum(proba * (1 - proba))


def entropy(x):
    _, counts = np.unique(x, return_counts=True)
    proba = counts / len(x)
    return -np.sum(proba * np.log2(proba))


def gain(left_y, right_y, criterion):
    y = np.concatenate((left_y, right_y))
    return criterion(y) - (criterion(left_y) * len(left_y) + criterion(right_y) * len(right_y)) / len(y)


class DecisionTreeNode:
    def __init__(self, dim, left, right):
        self.dim = dim
        self.left = left
        self.right = right


class DecisionTreeLeaf:
    def __init__(self, y):
        self.y = y
        labels, count = np.unique(y, return_counts=True)
        self.label = labels[np.argmax(count)]


class DecisionTree:
    def __init__(self, X, y, criterion="gini", max_depth=None, min_samples_leaf=1, max_features="auto"):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        if max_features == "auto":
            self.max_features = int(sqrt(len(X[0])))
        if criterion == "gini":
            self.criterion = gini
        if criterion == "entropy":
            self.criterion = entropy
        self.root = self.makeNode(np.array(list(zip(y, X))), 0)

    def gain(self, left_y, right_y, criterion):
        if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
            return -1
        else:
            return gain(left_y, right_y, criterion)

    def makeNode(self, X_y, depth):
        if depth == self.max_depth or len(X_y) < 2 * self.min_samples_leaf:
            return DecisionTreeLeaf(X_y[:, 0])
        all_features = np.array(list(range(len(X_y[0][1]))))
        np.random.shuffle(all_features)
        features = all_features[:self.max_features]
        gains = [self.gain(np.array([z[0] for z in filter(lambda x: x[1][d] == 0, X_y)]),
                           np.array([z[0] for z in filter(lambda x: x[1][d] == 1, X_y)]),
                           self.criterion) for d in features]
        dim = features[np.argmax(gains)]
        max_gain = max(gains)
        if max_gain <= 0:
            return DecisionTreeLeaf(X_y[:, 0])
        else:
            left = np.array(list(filter(lambda x: x[1][dim] == 0, X_y)))
            right = np.array(list(filter(lambda x: x[1][dim] == 1, X_y)))
            return DecisionTreeNode(dim, self.makeNode(left, depth + 1),
                                    self.makeNode(right, depth + 1))

    def find_x(self, tree, x):
        if isinstance(tree, DecisionTreeLeaf):
            return tree.label
        else:
            if x[tree.dim] == 0:
                return self.find_x(tree.left, x)
            else:
                return self.find_x(tree.right, x)

    def predict(self, X):
        return [self.find_x(self.root, x) for x in X]


class RandomForestClassifier:
    def __init__(self, criterion="gini", max_depth=None, min_samples_leaf=1, max_features="auto", n_estimators=10,
                 ratio=1.0):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        if max_features == "auto":
            self.max_features = int(sqrt(len(X[0])))
        self.n_estimators = n_estimators
        self.ratio = ratio
        self.X = None
        self.y = None
        self.trees = None
        self.oob = None
        self.m = None

    def importance(self):
        importance = np.array([0.0 for j in range(len(X[0]))])
        for n in range(self.n_estimators):
            print("importance:", n)
            X_oob = np.array([self.X[i] for i in self.oob[n]])
            y_oob = np.array([self.y[i] for i in self.oob[n]])
            err_obb = np.mean(self.trees[n].predict(X_oob) == y_oob)
            for j in range(len(X[0])):
                col_j = copy.copy(X_oob[:, j])
                np.random.shuffle(col_j)
                X_oob_j = copy.copy(X_oob)
                X_oob_j[:, j] = col_j
                err_j = np.mean(self.trees[n].predict(X_oob_j) == y_oob)
                importance[j] += err_obb - err_j
        return importance / self.n_estimators

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.m = int(self.ratio * len(X))

        def baggingTree(n):
            print("bagging:", n)
            randind = np.random.randint(len(X), size=self.m)
            tree = DecisionTree([X[i] for i in randind], [y[i] for i in randind], criterion=self.criterion,
                                min_samples_leaf=self.min_samples_leaf,
                                max_depth=self.max_depth, max_features=self.max_features)
            out_of_bag = np.array(list(filter(lambda x: x not in randind, range(len(X)))))
            return tree, out_of_bag

        trees_oob = np.array([baggingTree(n) for n in range(self.n_estimators)])
        self.trees = trees_oob[:, 0]
        self.oob = trees_oob[:, 1]

    def predict(self, X):
        preds = np.transpose(np.array([tree.predict(X) for tree in self.trees]))

        def best(predictions):
            labels, count = np.unique(predictions, return_counts=True)
            return labels[np.argmax(count)]

        return [best(y) for y in preds]


def feature_importance(rfc):
    return rfc.importance()


def most_important_features(importance, names, k=20):
    idicies = np.argsort(importance)[::-1][:k]
    return np.array(names)[idicies]


def print_most_important(importance, names, k=50):
    idicies = np.argsort(importance)[::-1][:k]
    for i in idicies:
        print(importance[i], names[i])


def synthetic_dataset(size):
    X = [(np.random.randint(0, 2), np.random.randint(0, 2), int(i % 6 == 3),
          int(i % 6 == 0), int(i % 3 == 2), np.random.randint(0, 2)) for i in range(size)]
    y = [i % 3 for i in range(size)]
    return np.array(X), np.array(y)


X, y_age, y_sex, features = read_dataset("vk.csv")
X_train, X_test, y_age_train, y_age_test, y_sex_train, y_sex_test = train_test_split(X, y_age, y_sex, train_size=0.9)
#X, y = synthetic_dataset(1000)

# forest = RandomForestClassifier(min_samples_leaf=1, n_estimators=10)
#
# start = time.time()
#
# forest.fit(X_train, y_age_train)
#
# print("fit", time.time() - start)
#
# start = time.time()
#
# print(np.mean(forest.predict(X_test) == y_age_test))
#
# print("predict", time.time() - start)
#
# start = time.time()
#
#
# imp = forest.importance()
# print(len(imp))
# print_most_important(imp, features)
#
# print("importance", time.time() - start)
#
# exit(0)

#cbc.fit(X, y)
#print("Accuracy:", np.mean(cbc.predict(X).reshape((len(y))) == y))
#print("Importance:", cbc.get_feature_importance())
cbc = CatBoostClassifier(iterations=100,
                         loss_function='MultiClass',
                         task_type="GPU"
                         )
cbc.fit(X_train, y_age_train)
print("Accuracy:", np.mean(cbc.predict(X_test).reshape((len(y_age_test))) == y_age_test))
print("Most important features:")
for i, name in enumerate(most_important_features(cbc.get_feature_importance(), features, 10)):
    print(str(i+1) + ".", name)

cbc.fit(X_train, y_sex_train)
print("Accuracy:", np.mean(cbc.predict(X_test).reshape((len(y_sex_test))) == y_sex_test))
print("Most important features:")
for i, name in enumerate(most_important_features(cbc.get_feature_importance(), features, 10)):
    print(str(i+1) + ".", name)
# tree = DecisionTree(X_train, y_train, max_depth=30, min_samples_leaf=10)

# forest = RandomForestClassifier(max_depth=30, min_samples_leaf=10, n_estimators=20)
# forest.fit(X_train, y_train)

# pred0 = tree.predict(X_test)

# pred = forest.predict(X_test)

# print(len(list(filter(lambda x: x[0] != x[1], zip(pred0, y_test)))) / len(y_test))
# print(len(list(filter(lambda x: x[0] != x[1], zip(pred, y_test)))) / len(y_test))

# print_most_important(forest.importance(), publics)
# print(most_important_features(forest.importance(), publics))
