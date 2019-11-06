import numpy as np
import copy
from sklearn.datasets import make_blobs, make_moons
import torch

class Module:
    def forward(self, x):
        raise NotImplementedError()

    def backward(self, d):
        raise NotImplementedError()

    def update(self, alpha):
        pass


class Linear(Module):
    def __init__(self, in_features, out_features):
        pass

    def forward(self, x):
        pass

    def backward(self, d):
        pass

    def update(self, alpha):
        pass


class ReLU(Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass

    def backward(self, d):
        pass


class Softmax(Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass

    def backward(self, d):
        pass