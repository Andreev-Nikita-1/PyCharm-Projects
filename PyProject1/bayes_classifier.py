from builtins import filter

from sklearn.model_selection import train_test_split
import numpy as np
import pandas
import random
import matplotlib.pyplot as plt
import matplotlib
import copy
import spacy
from nltk.stem.snowball import SnowballStemmer

import re


def read_dataset(filename):
    file = open(filename, encoding="utf-8")
    x = []
    y = []
    for line in file:
        cl, sms = re.split("^b['\"](ham|spam)['\"][,\t\s]+b['\"](.*)['\"]$", line)[1:3]
        x.append(sms)
        y.append(cl)
    return np.array(x, dtype=np.str), np.array(y, dtype=np.str)


def get_precision_recall_accuracy(y_pred, y_true):
    classes = np.unique(list(y_pred) + list(y_true))
    true_positive = dict((c, 0) for c in classes)
    true_negative = dict((c, 0) for c in classes)
    false_positive = dict((c, 0) for c in classes)
    false_negative = dict((c, 0) for c in classes)
    for c_pred, c_true in zip(y_pred, y_true):
        for c in classes:
            if c_true == c:
                if c_pred == c_true:
                    true_positive[c] = true_positive.get(c, 0) + 1
                else:
                    false_negative[c] = false_negative.get(c, 0) + 1
            else:
                if c_pred == c:
                    false_positive[c] = false_positive.get(c, 0) + 1
                else:
                    true_negative[c] = true_negative.get(c, 0) + 1
    precision = dict((c, true_positive[c] / (true_positive[c] + false_positive[c])) for c in classes)
    recall = dict((c, true_positive[c] / (true_positive[c] + false_negative[c])) for c in classes)
    accuracy = sum([true_positive[c] for c in classes]) / len(y_pred)
    return precision, recall, accuracy


def plot_precision_recall(X_train, y_train, X_test, y_test, bow_method, voc_sizes=range(4, 200, 5)):
    classes = np.unique(list(y_train) + list(y_test))
    precisions = dict([(c, []) for c in classes])
    recalls = dict([(c, []) for c in classes])
    accuracies = []
    for v in voc_sizes:
        bow = bow_method(X_train, voc_limit=v)
        X_train_transformed = bow.transform(X_train)
        X_test_transformed = bow.transform(X_test)
        classifier = NaiveBayes(0.001)
        classifier.fit(X_train_transformed, y_train)
        y_pred = classifier.predict(X_test_transformed)
        precision, recall, acc = get_precision_recall_accuracy(y_pred, y_test)
        for c in classes:
            precisions[c].append(precision[c])
            recalls[c].append(recall[c])
        accuracies.append(acc)

    def plot(x, ys, ylabel, legend=True):
        plt.figure(figsize=(12, 3))
        plt.xlabel("Vocabulary size")
        plt.ylabel(ylabel)
        plt.xlim(x[0], x[-1])
        plt.ylim(np.min(list(ys.values())) - 0.01, np.max(list(ys.values())) + 0.01)
        for c in ys.keys():
            plt.plot(x, ys[c], label="Class " + str(c))
        if legend:
            plt.legend()
        plt.tight_layout()
        plt.show()

    plot(voc_sizes, recalls, "Recall")
    plot(voc_sizes, precisions, "Precision")
    plot(voc_sizes, {"": accuracies}, "Accuracy", legend=False)


X, y = read_dataset("data/spam.csv")
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)


class NaiveBayes:
    def __init__(self, alpha):
        self.alpha = alpha  # Параметр аддитивной регуляризации

    def fit(self, X, y):
        self.classes = np.unique(y)
        sums = np.array([np.sum(X[y == l], axis=0) + self.alpha for l in self.classes])
        self.logprobas = np.log(sums / np.sum(sums, axis=1).reshape(-1, 1))
        self.class_logprobas = np.log(np.array([len(y[(y == l)]) for l in self.classes]) / len(y))

    def predict(self, X):
        return [self.classes[i] for i in np.argmax(self.log_proba(X), axis=1)]

    def log_proba(self, X):
        return np.array(
            [[self.class_logprobas[i] + np.sum(x * self.logprobas[i]) for i in range(len(self.classes))] for x in X])


# X = np.array([[5, 0, 0], [1, 1, 1], [5, 1, 1]])
# y = np.array(['a', 'b', 'a'])
# Xtest = np.array([[3, 1, 0], [1, 2, 1], [3, 2, 1]])
# NB = NaiveBayes(1)
# NB.fit(X, y)
# print(NB.predict(Xtest))


class BoW:
    def __init__(self, X, voc_limit=1000):
        bag = dict()
        for x in X:
            for w in re.split('[^a-zA-Z]', x.lower()):
                if w not in bag.keys():
                    bag[w] = 1
                else:
                    bag[w] += 1

        if '' in bag.keys():
            del bag['']
        self.bag = np.array(sorted(bag.items(), key=lambda x: x[1], reverse=True))[:voc_limit, 0]

    def transform(self, X):
        ans = []
        for x in X:
            words = list(re.split('[^a-zA-Z]', x.lower()))
            ans.append([words.count(self.bag[i]) for i in range(len(self.bag))])
        return np.array(ans)


class BowStem:
    def __init__(self, X, voc_limit=1000):
        self.stemmer = SnowballStemmer('english')
        bag = dict()
        for x in X:
            for w in re.split('[^a-zA-Z]', x.lower()):
                if w not in bag.keys():
                    bag[self.stemmer.stem(w)] = 1
                else:
                    bag[self.stemmer.stem(w)] += 1

        if '' in bag.keys():
            del bag['']
        self.bag = np.array(sorted(bag.items(), key=lambda x: x[1], reverse=True))[:voc_limit, 0]

    def transform(self, X):
        ans = []
        for x in X:
            words = list(re.split('[^a-zA-Z]', x.lower()))
            words = [self.stemmer.stem(w) for w in words]
            ans.append([words.count(self.bag[i]) for i in range(len(self.bag))])
        return np.array(ans)


bow = BoW(X_train, voc_limit=500)
X_train_bow = bow.transform(X_train)
X_test_bow = bow.transform(X_test)
predictor = NaiveBayes(0.001)
predictor.fit(X_train_bow, y_train)
get_precision_recall_accuracy(predictor.predict(X_test_bow), y_test)
plot_precision_recall(X_train, y_train, X_test, y_test, BoW)
